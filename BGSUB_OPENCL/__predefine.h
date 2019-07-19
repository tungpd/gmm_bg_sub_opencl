//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// This is the GPU implementation of Zivkovic's Background Subtraction algorithm.
// For more details:
//		"Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction"
//			Z.Zivkovic, F. van der Heijden, Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.
//and
//		"Recursive unsupervised learning of finite mixture models"
//			Z.Zivkovic, F.van der Heijden, IEEE Trans. on Pattern Analysis and Machine Intelligence, vol.26, no.5, pages 651-656, 2004.
//
//Copyright (c) 2012 Pham Duy Tung.
//
//
//NOTE: 
//		This program is free software: you can redistribute it and/or modify
//		it under the terms of the GNU General Public License as published by
//		the Free Software Foundation, either version 3 of the License, or
//		(at your option) any later version.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//
// Version: 0.007, Date: 03-April-2012.
//
// Contact information:
//		Email:		duytung88@gmail.com
//		website:	http://duytung88.appspot.com
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#ifndef __PREDEFINE_H__
#define __PREDEFINE_H__

#define __OPENCL__
#define __USE_OPENCV__

#ifdef __cplusplus
#define __EXTERN_C		extern "C"
#else
#define __EXTERN_C		
#endif

#if (defined WIN32)||(defined WIN64)
#define __EXPORTS			__declspec(dllexport)
#else
#define __EXPORTS
#endif

#define __IMPORTS			__EXTERN_C

#define __API			__EXTERN_C __EXPORTS

#define GMM_KERNEL_NAME		"gmmBgSub"

//define build options
#define GMM_DEFAULT_BUILD_OPTIONS		"-D "

#define GMM_NUM_WORK_GROUPS_PER_COMPUTE_UNIT	2

//Define Device types
#define GMM_DEVICE_TYPE_CPU				0x12312310
#define GMM_DEVICE_TYPE_CPU_INTEL		0x12312311
#define GMM_DEVICE_TYPE_CPU_AMD			0x12312312
#define GMM_DEVICE_TYPE_GPU				0x12312320
#define GMM_DEVICE_TYPE_GPU_NVIDIA		0x12312321
#define GMM_DEVICE_TYPE_GPU_AMD			0x12312322

#define __DEBUG__

#define __OVERLAP_COPY__				1
#define __GPU_TIMER__
//#define __PROC_MULTIFRAMES__
#endif