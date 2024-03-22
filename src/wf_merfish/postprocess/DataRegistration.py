"""
DataRegistration: Register qi2lab widefield MERFISH data using cross-correlation and optical flow

Shepherd 2024/01 - updates for qi2lab MERFISH file format v1.0
Shepherd 2023/09 - initial commit
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
from numpy.typing import NDArray
import zarr
import gc
from cmap import Colormap
import warnings
import SimpleITK as sitk
from numcodecs import blosc
from wf_merfish.postprocess._registration import compute_optical_flow, apply_transform, downsample_image, compute_rigid_transform, normalize_histograms
from clij2fft.richardson_lucy import richardson_lucy_nc, getlib
from spots3d import SPOTS3D

class DataRegistration:
    def __init__(self,
                 dataset_path: Union[str, Path],
                 overwrite_registered: bool = False,
                 perform_optical_flow: bool = False,
                 tile_idx: Optional[int] = None):
        """
        Retrieve and pre-process one tile from qi2lab 3D widefield zarr structure.
        Apply rigid and optical flow registration transformations if available.

        Parameters
        ----------
        dataset_path : Union[str, Path]
            Path to Zarr dataset
        tile_idx : int
            tile index to retrieve
        """

        self._dataset_path = dataset_path
        self._polyDT_dir_path = dataset_path / Path("polyDT")

        if tile_idx is None:
            tile_idx = 0
        self._tile_idx = tile_idx        
        self._parse_dataset()
        
        self._perform_optical_flow = perform_optical_flow
        self._data_raw = None
        self._rigid_xforms = None
        self._of_xforms = None
        self._has_registered_data = None
        self._has_rigid_registrations = False
        self._has_of_registrations = False
        self._compressor = blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.Blosc.BITSHUFFLE)
        blosc.set_nthreads(20)
        self._overwrite_registered = overwrite_registered

    # -----------------------------------
    # property access for class variables
    # -----------------------------------
    @property
    def dataset_path(self):
        if self._dataset_path is not None:
            return self._dataset_path
        else:
            warnings.warn('Dataset path not defined.',UserWarning)
            return None

    @dataset_path.setter
    def dataset_path(self,new_dataset_path: Union[str, Path]):
        del self.dataset_path
        self._dataset_path = new_dataset_path
        self._dataset = zarr.open_group(self._dataset_path)

    @dataset_path.deleter
    def dataset_path(self):
        if self._dataset_path is not None:
            del self._dataset_path
            del self._dataset
            self._dataset_path = None
            self._dataset = None

    @property
    def tile_idx(self):
        if self._tile_idx is not None:
            tile_idx = self._tile_idx
            return tile_idx
        else:
            warnings.warn('Tile coordinates not defined.',UserWarning)
            return None

    @tile_idx.setter
    def tile_idx(self,new_tile_idx: int):
        self._has_registrations = False
        self._tile_idx = new_tile_idx
        self._tile_id = 'tile'+str(self._tile_idx).zfill(4)

    @tile_idx.deleter
    def tile_idx(self):
        if self._tile_idx is not None:
            del self.data_raw
            del self.data_registered
            del self.rigid_xforms
            del self.of_xforms
            del self._tile_idx
            self._has_registrations = False
            self._tile_idx = None
            self._tile_id = None

    @property
    def data_raw(self):
        if self._data_raw is not None:
            return self._data_raw
        else:
            warnings.warn('Data not loaded.',UserWarning)
            return None
    
    @data_raw.deleter
    def data_raw(self):
        if self._data_raw is not None:
            del self._data_raw
            gc.collect()
            self._data_raw = None

    @property
    def data_registered(self):
        return self._data_registered
    
    @data_registered.deleter
    def data_registered(self):
        del self._data_registered
        gc.collect()
        self._data_registered= None

    @property
    def rigid_xforms(self):
        if self._total_xform is not None:
            return self._total_xform
        else:
            warnings.warn('Rigid transforms not loaded.',UserWarning)
            return None
    
    @rigid_xforms.deleter
    def rigid_xforms(self):
        if self._total_xform is not None:
            del self._total_xform
            gc.collect()
            self._total_xform = None

    @property
    def perform_optical_flow(self):
        return self._perform_optical_flow
    
    @perform_optical_flow.setter
    def perform_optical_flow(self, new_perform_optical_flow: bool):
        self._perform_optical_flow = new_perform_optical_flow

    @property
    def of_xforms(self):
        if self._of_xforms is not None:
            return self._of_xforms
        else:
            warnings.warn('Optical flow fields not loaded.',UserWarning)
            return None
    
    @of_xforms.deleter
    def of_xforms(self):
        if self._of_xforms is not None:
            del self._of_xforms
            gc.collect()
            self._of_xforms = None

    def _parse_dataset(self):
        """
        Parse dataset to discover number of tiles, number of rounds, and voxel size.
        """

        self._tile_ids = sorted([entry.name for entry in self._polyDT_dir_path.iterdir() if entry.is_dir()],
                                key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))
        self._num_tiles = len(self._tile_ids)
        self._tile_id = self._tile_ids[self._tile_idx]

        current_tile_dir_path = self._polyDT_dir_path / Path(self._tile_id)
        self._round_ids = sorted([entry.name.split('.')[0]  for entry in current_tile_dir_path.iterdir() if entry.is_dir()],
                                 key=lambda x: int(x.split('round')[1].split('.zarr')[0]))
        round_id = self._round_ids[0]
        current_round_zarr_path = current_tile_dir_path / Path(round_id + ".zarr")
        current_round = zarr.open(current_round_zarr_path,mode='r')

        self._voxel_size = np.asarray(current_round.attrs['voxel_zyx_um'], dtype=np.float32)
        
        calibrations_dir_path = self._dataset_path / Path('calibrations.zarr')
        calibrations_zarr = zarr.open(calibrations_dir_path,mode='r')
        self._psfs = np.asarray(calibrations_zarr['psf_data'],dtype=np.uint16)

        del current_round, calibrations_zarr
        gc.collect()

    def load_raw_data(self):
        """
        Load raw data across rounds.
        """

        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None

        data_raw = []
        stage_positions = []
        
        for round_id in self._round_ids:
            current_round_zarr_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
            current_round = zarr.open(current_round_zarr_path,mode='r')
            gain = float(current_round.attrs['gain'])
            data_raw.append(np.asarray(current_round['raw_data']).astype(np.uint16))
            stage_positions.append(np.asarray(current_round.attrs['stage_zyx_um'], dtype=np.float32))

        self._data_raw = np.stack(data_raw, axis=0)
        self._stage_positions = np.stack(stage_positions,axis=0)
        del data_raw, stage_positions, current_round
        gc.collect()

    def load_registered_data(self,
                             readouts=False):
        """
        If available, load registered data across rounds.
        """
        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None
        
        data_registered = []
        
        if not(readouts):
            try:
                for round_id in self._round_ids:
                    current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
                    current_round = zarr.open(current_round_path,mode='r')
                    data_registered.append(np.asarray(current_round["registered_decon_data"], dtype=np.uint16))
                            
                self._data_registered = np.stack(data_registered,axis=0)
                del data_registered, current_round
                gc.collect()
                self._has_registered_data = True

            except Exception:
                warnings.warn('Generate registered data first.',UserWarning)
                return None
        else:
            try:
                readout_dir_path = self._dataset_path / Path('readouts')
                tile_ids = sorted([entry.name for entry in readout_dir_path.iterdir() if entry.is_dir()],
                                  key=lambda x: int(x.split('tile')[1].split('.zarr')[0]))
                tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
                self._bit_ids = sorted([entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()],
                                       key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))
                
                current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(self._round_ids[0] + ".zarr")
                current_round = zarr.open(current_round_path,mode='r')
                data_registered.append(np.asarray(current_round["registered_decon_data"], dtype=np.uint16))
                
                for bit_id in self._bit_ids:
                    tile_dir_path = readout_dir_path / Path(tile_ids[self._tile_idx])
                    bit_dir_path = tile_dir_path / Path(bit_id)
                    current_bit = zarr.open(bit_dir_path,mode='r')
                    data_registered.append(np.asarray(current_bit["registered_decon_data"], dtype=np.uint16))
                    
                self._data_registered = np.stack(data_registered,axis=0)
                del data_registered, current_round
                gc.collect()
                self._has_registered_data = True
            except Exception:
                warnings.warn('Generate registered data first.',UserWarning)
                return None
 
    def generate_registrations(self):
        """
        Generate registration transforms using reference channel.
        Use cross-correlation translation in stages followed by optional optical flow refinement.
        """

        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None
        
        if self._data_raw is None:
            self.load_raw_data()
            
        lib = getlib()
            
        ref_image_decon = richardson_lucy_nc(np.asarray(self._data_raw[0,:]),
                                            psf=self._psfs[0,:],
                                            numiterations=40,
                                            regularizationfactor=.0001,
                                            lib=lib)
                
        current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(self._round_ids[0] + ".zarr")
        current_round = zarr.open(current_round_path,mode='a')
        
        try:
            data_decon_zarr = current_round.zeros('decon_data',
                                            shape=ref_image_decon.shape,
                                            chunks=(1,ref_image_decon.shape[1],ref_image_decon.shape[2]),
                                            compressor=self._compressor,
                                            dtype=np.uint16)
            data_reg_zarr = current_round.zeros('registered_decon_data',
                                            shape=ref_image_decon.shape,
                                            chunks=(1,ref_image_decon.shape[1],ref_image_decon.shape[2]),
                                            compressor=self._compressor,
                                            dtype=np.uint16)
        except Exception:
            data_decon_zarr = current_round['decon_data']
            data_reg_zarr = current_round['registered_decon_data']
        
        data_decon_zarr[:] = ref_image_decon
        data_reg_zarr[:] = ref_image_decon
                    
        ref_image_sitk = sitk.GetImageFromArray(ref_image_decon.astype(np.float32))
        
        del current_round_path, current_round, data_reg_zarr
        gc.collect()

        for r_idx, round_id in enumerate(self._round_ids[1:]):
            r_idx = r_idx + 1
            current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
            current_round = zarr.open(current_round_path,mode='a')
            psf_idx = current_round.attrs["psf_idx"]
                
            mov_image_decon = richardson_lucy_nc(self._data_raw[r_idx,:],
                                            psf=self._psfs[psf_idx,:],
                                            numiterations=40,
                                            regularizationfactor=.0001,
                                            lib=lib)

            mov_image_sitk = sitk.GetImageFromArray(mov_image_decon.astype(np.float32))
                            
            downsample_factor = 4
            if downsample_factor > 1:
                ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
            else:
                ref_ds_image_sitk = ref_image_sitk
                mov_ds_image_sitk = mov_image_sitk
                
            _, initial_xy_shift = compute_rigid_transform(ref_ds_image_sitk, 
                                                            mov_ds_image_sitk,
                                                            use_mask=False,
                                                            downsample_factor=downsample_factor,
                                                            projection='z')
            
            intial_xy_transform = sitk.TranslationTransform(3, initial_xy_shift)

            mov_image_sitk = apply_transform(ref_image_sitk,
                                                mov_image_sitk,
                                                intial_xy_transform)
            
            del ref_ds_image_sitk
            gc.collect()
            
            downsample_factor = 4
            if downsample_factor > 1:
                ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
            else:
                ref_ds_image_sitk = ref_image_sitk
                mov_ds_image_sitk = mov_image_sitk
                
            _, intial_z_shift = compute_rigid_transform(ref_ds_image_sitk, 
                                                        mov_ds_image_sitk,
                                                        use_mask=False,
                                                        downsample_factor=downsample_factor,
                                                        projection='search')
            
            intial_z_transform = sitk.TranslationTransform(3, intial_z_shift)

            mov_image_sitk = apply_transform(ref_image_sitk,
                                            mov_image_sitk,
                                            intial_z_transform)
            
            del ref_ds_image_sitk
            gc.collect()
            
            downsample_factor = 4
            if downsample_factor > 1:
                ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
            else:
                ref_ds_image_sitk = ref_image_sitk
                mov_ds_image_sitk = mov_image_sitk
                                
            _, xyz_shift_4x = compute_rigid_transform(ref_ds_image_sitk, 
                                                        mov_ds_image_sitk,
                                                        use_mask=False,
                                                        downsample_factor=downsample_factor,
                                                        projection=None)
    
            
            final_xyz_shift = np.asarray(initial_xy_shift) + np.asarray(intial_z_shift) + np.asarray(xyz_shift_4x)                        
            current_round.attrs["rigid_xform_xyz_um"] = final_xyz_shift.tolist()
            
            xyz_transform_4x = sitk.TranslationTransform(3, xyz_shift_4x)
            mov_image_sitk = apply_transform(ref_image_sitk,
                                                    mov_image_sitk,
                                                    xyz_transform_4x)
            del ref_ds_image_sitk
            gc.collect()
            
            if self._perform_optical_flow:
                    
                downsample_factor = 4
                if downsample_factor > 1:
                    ref_ds_image_sitk = downsample_image(ref_image_sitk, downsample_factor)
                    mov_ds_image_sitk = downsample_image(mov_image_sitk, downsample_factor)
                else:
                    mov_ds_image_sitk = mov_image_sitk

                ref_ds_image = sitk.GetArrayFromImage(ref_ds_image_sitk).astype(np.float32)
                mov_ds_image = sitk.GetArrayFromImage(mov_ds_image_sitk).astype(np.float32)
                del ref_ds_image_sitk, mov_ds_image_sitk
                gc.collect()
                
                of_xform_4x = compute_optical_flow(ref_ds_image,mov_ds_image)
                del ref_ds_image, mov_ds_image
                gc.collect()

                try:
                    of_xform_zarr = current_round.zeros('of_xform_4x',
                                                    shape=of_xform_4x.shape,
                                                    chunks=(1,1,of_xform_4x.shape[2],of_xform_4x.shape[3]),
                                                    compressor=self._compressor,
                                                    dtype=np.float32)
                except Exception:
                    of_xform_zarr = current_round['of_xform_4x']
                
                of_xform_zarr[:] = of_xform_4x
                
                of_4x_sitk = sitk.GetImageFromArray(of_xform_4x.transpose(1, 2, 3, 0).astype(np.float64),
                                                        isVector = True)
                interpolator = sitk.sitkLinear
                identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                optical_flow_sitk = sitk.Resample(of_4x_sitk, mov_image_sitk, identity_transform, interpolator,
                                            0, of_4x_sitk.GetPixelID())
                displacement_field = sitk.DisplacementFieldTransform(optical_flow_sitk)
                del of_4x_sitk, of_xform_4x
                gc.collect()
                
                # apply optical flow 
                mov_image_sitk = sitk.Resample(mov_image_sitk,displacement_field)
                data_registered = sitk.GetArrayFromImage(mov_image_sitk).astype(np.uint16)

                del optical_flow_sitk, displacement_field
                gc.collect()
            else:
                data_registered = sitk.GetArrayFromImage(mov_image_sitk).astype(np.uint16)
                
            try:
                data_decon_zarr = current_round.zeros('decon_data',
                                                    shape=ref_image_decon.shape,
                                                    chunks=(1,ref_image_decon.shape[1],ref_image_decon.shape[2]),
                                                    compressor=self._compressor,
                                                    dtype=np.uint16)
                data_reg_zarr = current_round.zeros('registered_decon_data',
                                                shape=data_registered.shape,
                                                chunks=(1,data_registered.shape[1],data_registered.shape[2]),
                                                compressor=self._compressor,
                                                dtype=np.uint16)
            except Exception:
                data_decon_zarr = current_round['decon_data']
                data_reg_zarr = current_round['registered_decon_data']
            
            data_decon_zarr[:] = mov_image_decon
            data_reg_zarr[:] = data_registered
            
            del mov_image_decon, data_registered, mov_image_sitk
            gc.collect()
        
        del ref_image_sitk
        gc.collect()
                        
    def apply_registration_to_bits(self):
        """
        Generate registered data and save to zarr.
        """
       
        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None
               
        if not(self._has_rigid_registrations):
            self.load_rigid_registrations()
            if not(self._has_rigid_registrations):
                raise Exception("Create rigid registrations first.")

        if self._perform_optical_flow and not(self._has_of_registrations):
            self.load_opticalflow_registrations()
            if not(self._has_of_registrations):
                raise Exception("Create vector field registrations first.")
            

        readout_dir_path = self._dataset_path / Path('readouts')
        tile_dir_path = readout_dir_path / Path(self._tile_id)
        bit_ids = sorted([entry.name for entry in tile_dir_path.iterdir() if entry.is_dir()],
                         key=lambda x: int(x.split('bit')[1].split('.zarr')[0]))
                
        tile_dir_path = readout_dir_path / Path(self._tile_id)
        
        round_list = []
        for bit_id in bit_ids:
            bit_dir_path = tile_dir_path / Path(bit_id)
            current_bit_channel = zarr.open(bit_dir_path,mode='a')      
            round_list.append(int(current_bit_channel.attrs['round']))
            
        round_list = np.array(round_list)
        ref_idx = int(np.argwhere(round_list==0)[0])
        print('found ref. index = ' + str(ref_idx))
        
        ref_bit_id = bit_ids[ref_idx]
        bit_dir_path = tile_dir_path / Path(ref_bit_id)
        reference_bit_channel = zarr.open(bit_dir_path,mode='a')      
    
        ref_dog_bit_sitk = sitk.GetImageFromArray(reference_bit_channel['raw_data'].astype(np.float32))
            
        del reference_bit_channel
        gc.collect()
        
        for bit_id in bit_ids:
            bit_dir_path = tile_dir_path / Path(bit_id)
            current_bit_channel = zarr.open(bit_dir_path,mode='a')      
            r_idx = int(current_bit_channel.attrs['round'])
            psf_idx = int(current_bit_channel.attrs['psf_idx'])
            gain = float(current_bit_channel.attrs['gain'])
            em_wvl = float(current_bit_channel.attrs['emission_um'])
            
            
            spots_factory = SPOTS3D.SPOTS3D(data=current_bit_channel['raw_data'].astype(np.uint16),
                                            psf=self._psfs[psf_idx,:],
                                            microscope_params={'na': 1.35,
                                                               'ri': 1.51,
                                                               'theta': 0},
                                            metadata={'pixel_size' : self._voxel_size[2],
                                                    'scan_step' : self._voxel_size[0],
                                                    'wvl': em_wvl},
                                            decon_params={'iterations' : 40,
                                                          'tv_tau' : .0001},
                                            DoG_filter_params=None)
            
            spots_factory.run_deconvolution()
            spots_factory.run_DoG_filter()
            
            decon_image = spots_factory.decon_data.copy().astype(np.uint16)
            dog_image = spots_factory.dog_data.copy().astype(np.float32)
            dog_image[dog_image<0.]=0.0
              
            del spots_factory
            gc.collect()
               
            if r_idx > 0:
                polyDT_tile_round_path = self._dataset_path / Path('polyDT') / Path(self._tile_id) / Path('round'+str(r_idx).zfill(3)+'.zarr')
                current_polyDT_channel = zarr.open(polyDT_tile_round_path,mode='r')
                
                rigid_xform_xyz_um = np.asarray(current_polyDT_channel.attrs['rigid_xform_xyz_um'],dtype=np.float32)
                shift_xyz = [float(i) for i in rigid_xform_xyz_um]
                xyx_transform = sitk.TranslationTransform(3, shift_xyz)           
                
                if self._perform_optical_flow:
                    of_xform_4x_xyz = np.asarray(current_polyDT_channel['of_xform_4x'],dtype=np.float32)
                    of_4x_sitk = sitk.GetImageFromArray(of_xform_4x_xyz.transpose(1, 2, 3, 0).astype(np.float64),
                                                                isVector = True)
                    interpolator = sitk.sitkLinear
                    identity_transform = sitk.Transform(3, sitk.sitkIdentity)
                    optical_flow_sitk = sitk.Resample(of_4x_sitk, ref_dog_bit_sitk, identity_transform, interpolator,
                                                    0, of_4x_sitk.GetPixelID())
                    displacement_field = sitk.DisplacementFieldTransform(optical_flow_sitk)
                    del rigid_xform_xyz_um, shift_xyz, of_xform_4x_xyz, of_4x_sitk, optical_flow_sitk
                    gc.collect()
                    
                decon_bit_image_sitk = apply_transform(ref_dog_bit_sitk,
                                                     sitk.GetImageFromArray(decon_image),
                                                     xyx_transform)

                dog_bit_image_sitk = apply_transform(ref_dog_bit_sitk,
                                                     sitk.GetImageFromArray(dog_image),
                                                     xyx_transform)
                
                if self._perform_optical_flow:

                    decon_bit_image_sitk = sitk.Resample(decon_bit_image_sitk,displacement_field)
                    dog_bit_image_sitk = sitk.Resample(dog_bit_image_sitk,displacement_field)
                
                data_decon_registered = sitk.GetArrayFromImage(decon_bit_image_sitk).astype(np.float32)
                data_dog_registered = sitk.GetArrayFromImage(dog_bit_image_sitk).astype(np.float32)
                del decon_bit_image_sitk, dog_bit_image_sitk, displacement_field
                gc.collect()
                
            else:
                data_decon_registered = decon_image.copy()
                data_dog_registered = dog_image.copy()
                gc.collect()
                
            data_decon = decon_image.copy()
            data_dog = dog_image.copy()
            
            del decon_image, dog_image
            
            try:
                data_decon_zarr = current_bit_channel.zeros('decon_data',
                                                        shape=data_decon.shape,
                                                        chunks=(1,data_decon.shape[1],data_decon.shape[2]),
                                                        compressor=self._compressor,
                                                        dtype=np.uint16)
                data_dog_zarr = current_bit_channel.zeros('dog_data',
                                                        shape=data_dog.shape,
                                                        chunks=(1,data_dog.shape[1],data_dog.shape[2]),
                                                        compressor=self._compressor,
                                                        dtype=np.float32)
                data_decon_reg_zarr = current_bit_channel.zeros('registered_decon_data',
                                                        shape=data_decon_registered.shape,
                                                        chunks=(1,data_decon_registered.shape[1],data_decon_registered.shape[2]),
                                                        compressor=self._compressor,
                                                        dtype=np.uint16)
                data_dog_reg_zarr = current_bit_channel.zeros('registered_dog_data',
                                                        shape=data_dog_registered.shape,
                                                        chunks=(1,data_dog_registered.shape[1],data_dog_registered.shape[2]),
                                                        compressor=self._compressor,
                                                        dtype=np.float32)
            except Exception:
                data_decon_zarr = current_bit_channel['decon_data']
                data_dog_zarr = current_bit_channel['dog_data']
                data_decon_reg_zarr = current_bit_channel['registered_decon_data']
                data_dog_reg_zarr = current_bit_channel['registered_dog_data']
                
            data_decon_zarr[:] = data_decon.astype(np.uint16)
            data_dog_zarr[:] = data_dog
            data_decon_reg_zarr[:] = data_decon_registered.astype(np.uint16)
            data_dog_reg_zarr[:] = data_dog_registered
            del data_dog_registered, data_decon_registered
            gc.collect()
            
        del data_decon, data_dog
        gc.collect()

    def load_rigid_registrations(self):
        """
        Load rigid registrations.
        """

        if self._tile_idx is None:
            print('Set tile position first.')
            print('e.g. DataLoader.tile_idx = 0')
            return None

        rigid_xform = []
        self._has_rigid_registrations = False

        try:
            for round_id in self._round_ids[1:]:
                current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
                current_round = zarr.open(current_round_path,mode='r')
                
                rigid_xform.append(np.asarray(current_round.attrs['rigid_xform_xyz_um'],dtype=np.float32))
            self._rigid_xforms = np.stack(rigid_xform,axis=0)
            self._has_rigid_registrations = True
            del rigid_xform
            gc.collect()
        except Exception:
            self._rigid_xforms = None
            self._has_rigid_registrations = False

    def load_opticalflow_registrations(self):
        """"
        Load optical flow registrations.
        """

        of_xform = []
        self._has_of_registrations = False

        try:
            for round_id in self._round_ids[1:]:
                current_round_path = self._polyDT_dir_path / Path(self._tile_id) / Path(round_id + ".zarr")
                current_round = zarr.open(current_round_path,mode='r')
                
                of_xform.append(np.asarray(current_round['of_xform_4x'],dtype=np.float32))
                self._of_xforms = np.stack(of_xform,axis=0)
            self._has_of_registrations = True
            del of_xform
            gc.collect()
        except Exception:
            self._of_xforms = None
            self._has_of_registrations = False

    # def export_tiled_tiffs(self,
    #                        tile_size: int = 425,
    #                        z_downsample: int = 2):
    #     """
    #     Export all RNA rounds, in codebook order, as small tiffs
    #     """
        
    #     import tifffile
    #     from skimage.transform import downscale_local_mean
    #     from functools import partial
    #     import json

    #     def write_tiled_tiffs(data_tile,
    #                           z_downsample: int,
    #                           stage_position: Sequence[float],
    #                           output_path: Union[str,Path],
    #                           block_info: dict = None):
    #         """
    #         Function to process and write tiff tiles and associated metadata.

    #         Parameters
    #         ----------
    #         data_tile: np.ndarray
    #             tile from map_overlap
    #         z_dowsample: int
    #             z downsampling factor
    #         stage_position: Sequene[float]
    #             zyx stage positions
    #         output_path: Union[str,Path]
    #             where to place the image
    #         block_info: dict
    #             block metadata from map_overlap

    #         Returns
    #         -------
    #         temp: np.ndarray
    #             fake image to make sure somethign is
    #             returned to map_overlap.
    #             TO DO: improve handling this so the code doesn't error
    #         """
            
    #         # helpful diagnostic code for checking tile metadata
    #         # print("========================")
    #         # print(block_info[0]['array-location'])
    #         # print(block_info[0]['chunk-location'])
    #         # print("========================")

    #         x_min = block_info[0]['array-location'][3][0] 
    #         x_max = block_info[0]['array-location'][3][1]
    #         y_min = block_info[0]['array-location'][2][0]
    #         y_max = block_info[0]['array-location'][2][1]
    #         z_min = block_info[0]['array-location'][1][0]
    #         z_max = block_info[0]['array-location'][1][1]

    #         x = block_info[0]['chunk-location'][3]
    #         y = block_info[0]['chunk-location'][2]
    #         z = block_info[0]['chunk-location'][1]

    #         stage_x = np.round(float(stage_position[0] + .115*(x_max-x_min)/2),2)
    #         stage_y = np.round(float(stage_position[1] + .115*(y_max-y_min)/2),2)
    #         stage_z = np.round(float(stage_position[2] + .230),2)

    #         downsample = downscale_local_mean(data_tile,(1,z_downsample,1,1),cval=0.0).astype(np.uint16)
    #         file_path = output_path / Path('subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.tiff')
    #         metadata_path_json = output_path / Path('subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.json')
    #         metadata_path_txt = output_path / Path('subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.txt')
    #         tile_metadata = {'subtile_id': 'subtile_x'+str(x).zfill(4)+'_y'+str(y).zfill(4)+'.tiff',
    #                          'start_x_pixel': x_min,
    #                          'end_x_pixel': x_max,
    #                          'start_y_pixel': y_min,
    #                          'end_y_pixel': y_max,
    #                          'start_z_pixel': z_min+1,
    #                          'end_z_pixel': z_max,
    #                         'stage_x': stage_x,
    #                         'stage_y': stage_y,
    #                         'stage_z': stage_z}
            
    #         tifffile.imwrite(output_path/file_path,
    #                          downsample[:,1:,:,:])
            
    #         with metadata_path_json.open('w') as f:
    #             json.dump(tile_metadata,f)

    #         df = pd.DataFrame(tile_metadata, index=[0]).T
    #         df.to_csv(metadata_path_txt, sep='\t', header=False)

    #         return np.array([1,1,1,1],dtype=np.uint8)
        
    #     # setup paths, metadata, codebook
    #     path_parts = self._dataset_path.parts
    #     tile_path = Path('x'+str(self._tile_coordinates[0]).zfill(3)\
    #                      +'_y'+str(self._tile_coordinates[1]).zfill(3)\
    #                      +'_z'+str(self._tile_coordinates[2]).zfill(3))
    #     output_path = Path(path_parts[0]) / Path(path_parts[1]) / Path(path_parts[2]) / Path("tiled_tiffs") / tile_path
    #     output_path.mkdir(parents=True, exist_ok=True)
    #     data_path = output_path / Path("data")
    #     data_path.mkdir(parents=True, exist_ok=True)
    #     metadata_path = output_path
    #     codebook_path_json = metadata_path / Path('codebook.json')
    #     codebook_path_txt = metadata_path / Path('codebook.txt')
    #     metadata_path_json = metadata_path / Path('exp_metadata.json')
    #     metadata_path_txt = metadata_path / Path('exp_metadata.txt')

    #     # generate registered data
    #     self.generate_registered_data()
                     
    #     # determine number of blocks in scan direction for tile size
    #     y_crop_index_max = (self.data_registered.shape[2]-1) // 425 * 425

    #     # reorder registered data in codebook order
    #     ordered_data = da.zeros((16,
    #                              self.data_registered.shape[1],
    #                              y_crop_index_max,
    #                              self.data_registered.shape[3]),
    #                             dtype=self.data_registered.dtype)
    #     idx = 0
    #     for key in self.exp_order:
    #         ordered_data[key-1,:] = self.data_registered[idx,:,0:y_crop_index_max,:]
    #         idx += 1
            
    #     ordered_data = da.rechunk(ordered_data,chunks=(-1,-1,tile_size,tile_size))

    #     exp_metadata = {'exp_name': str(output_path.parts[1]),
    #                     'tile_position_x': int(self.tile_coordinates['x']),
    #                     'tile_position_y': int(self.tile_coordinates['y']),
    #                     'tile_position_z': int(self.tile_coordinates['z']),
    #                     'tile_size_xy_pixel': int(tile_size),
    #                     'y_crop_begin_pixel': 0,
    #                     'y_crop_end_pixel':  int(y_crop_index_max),
    #                     'codebook_name': str(codebook_path.name),
    #                     'na': float(1.35),
    #                     'pixel_size_xy_um': float(0.115),
    #                     'pixel_size_z_um': float(0.115*z_downsample),
    #                     'slab_height_above_coverslip_um' : float(30.0*self.tile_coordinates['z']),
    #                     'deskewed': True,
    #                     'registered': True,
    #                     'deconvolved': False,
    #                     'DoG_filter': False}

    #     # write codebook and metadata
    #     with codebook_path_json.open('w') as f:
    #         json.dump(self.codebook,f)

    #     with metadata_path_json.open('w') as f:
    #         json.dump(exp_metadata,f)

    #     tile_loader._codebook.to_csv(codebook_path_txt,sep='\t',header=False,index=False)

    #     df = pd.DataFrame(exp_metadata, index=[0]).T
    #     df.to_csv(metadata_path_txt, sep='\t', header=False)

    #     # call map_blocks (or map_overlap) with requested tile_size, z downsampling, and tile overlap
    #     tile_writer_da = partial(write_tiled_tiffs,
    #                              z_downsample=z_downsample,
    #                              stage_position=self._stage_positions[0],
    #                              output_path=data_path)
    #     dask_writer = da.map_blocks(tile_writer_da,
    #                                 ordered_data,
    #                                 meta=np.array((), dtype=np.uint8))
        
    #     # use try to catch map_block dimension issue
    #     print('Generating tiled tiffs')
    #     print('----------------------')
    #     try:
    #         with TqdmCallback(desc="Subtiles"):
    #             dask_writer.compute(num_workers=8)
    #     except:
    #         pass

    #     del ordered_data, dask_writer, self._data_registered
    #     gc.collect()

    def _create_figure(self,
                       readouts=False):
        """
        Generate napari figure for debugging
        """
        import napari
        from qtpy.QtWidgets import QApplication

        def on_close_callback():
            viewer.layers.clear()
            gc.collect()
        
        viewer = napari.Viewer()
        app = QApplication.instance()

        app.lastWindowClosed.connect(on_close_callback)
        if not(readouts):
            viewer.window._qt_window.setWindowTitle(self._tile_ids[self._tile_idx] + ' alignment aross rounds')
        else:
            viewer.window._qt_window.setWindowTitle(self._tile_ids[self._tile_idx] + ' alignment aross bits')
        
        colormaps = [Colormap('cmap:white'),
                     Colormap('cmap:cyan'),
                     Colormap('cmap:yellow'),
                     Colormap('cmap:red'),
                     Colormap('cmap:green'),
                     Colormap('chrisluts:OPF_Fresh'),
                     Colormap('chrisluts:OPF_Orange'),
                     Colormap('chrisluts:OPF_Purple'),
                     Colormap('chrisluts:BOP_Blue'),
                     Colormap('chrisluts:BOP_Orange'),
                     Colormap('chrisluts:BOP_Purple'),
                     Colormap('cmap:cyan'),
                     Colormap('cmap:yellow'),
                     Colormap('cmap:red'),
                     Colormap('cmap:green'),
                     Colormap('chrisluts:OPF_Fresh'),
                     Colormap('chrisluts:OPF_Orange'),
                     Colormap('cmap:magenta')]
        
        if not(readouts):
            for idx in range(len(self._round_ids)):
                viewer.add_image(data=self._data_registered[idx],
                                name=self._round_ids[idx],
                                scale=self._voxel_size,
                                blending='additive',
                                colormap=colormaps[idx].to_napari(),
                                contrast_limits=[200,10000])
        else:
            viewer.add_image(data=self._data_registered[0],
                            name='polyDT',
                            scale=self._voxel_size,
                            blending='additive')
            for idx in range(len(self._bit_ids)):
                viewer.add_image(data=self._data_registered[idx+1],
                                name=self._bit_ids[idx],
                                scale=self._voxel_size,
                                blending='additive',
                                colormap=colormaps[idx].to_napari())

        napari.run()