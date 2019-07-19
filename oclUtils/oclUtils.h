/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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



#ifndef __CL_Tools_H
#define __CL_Tools_H

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#ifdef __cplusplus
#include <fstream>
#include <iostream>
#endif

#ifdef _WIN32
#include <CL\cl.h>
#include <Windows.h>
#else
#include <OpenCL\cl.h>
#endif

//#define __SSE2__

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __cplusplus
#ifndef OCL_BOOL
#define OCL_BOOL					bool
#define OCL_TRUE					true
#define OCL_FALSE					false
#endif
#else
#ifndef
#define OCL_BOOL					int
#define OCL_TRUE					1
#define OCL_FALSE					0
#endif
#endif

#ifndef OCL_EXTERN_C
#ifdef __cplusplus
#define OCL_EXTERN_C				extern "C"
#else
#define OCL_EXTERN_C
#endif
#endif

#ifndef OCL_INLINE
#ifdef __cplusplus
#define OCL_INLINE					inline
#elif (defined WIN32 || defined WIN64) && !(defined __GNUC__)
#define OCL_INLINE					__inline
#else
#define OCL_INLINE					static
#endif
#endif

#define OCL_INTERNAL				static

#ifndef _WIN32
	typedef uint64_t	memsize_t;
#else
typedef unsigned __int64 memsize_t;
#endif


//Max Min operators without jump
#ifndef OCL_MAX
//we have a^a^b==b, a^0==a for all a, b
//and (a^b)&(a>b - 1)==0 if a>b, ==1 if otherwise
//so a^((a^b)&(a>b - 1)) will return maximum of a,b
#define OCL_MAX(a, b)											((a)^(((a)^(b))&((a)>(b) - 1)))
#endif

#ifndef OCL_MIN
//we have a^a^b==b, a^0==a for all a, b
//and (a^b)&(a>b - 1)==0 if a>b, ==1 if otherwise
//so a^((a^b)&(a<b - 1)) will return minimum of a,b
#define OCL_MIN(a, b)											((a)^(((a)^(b))&((a)<(b) - 1)))	
#endif

#define OCL_SWAP(a, b, t)										t=a; a=b; b=t	

//return true if n is out of range
#define OCL_CHECK_OUT_OF_RANGE(n, min, max)						(((n)<(min))||((n)>(max)))
//define int max
#define OCL_INT_MAX												2147483647
//define unsigned int max
#define OCL_UINT_MAX											0xffffffff

#define oclIsPow2(n)											((n)&(n-1)==0)

//define max string lengh
#define OCL_STR_MAX_LEN											1024

#define OCL_IMAGE_WIDTH_MAX										1024
#define OCL_IMAGE_WIDTH_MIN										0
#define OCL_IMAGE_HEIGHT_MAX									1024
#define OCL_IMAGE_HEIGHT_MIN									0
#define OCL_IMAGE_PPM_MAX_PIX_VAL								255
#define OCL_IMAGE_PPM_MIN_PIX_VAL								0

#define OCL_DATA_MIN_SIZE										1
#define OCL_DATA_MAX_SIZE										(OCL_INT_MAX/4)

#ifndef OCL_CLAMP
#define OCL_CLAMP(a, b, c)										(MIN(MAX(a, b), c))
#endif

#ifndef OCL_ERROR
#define OCL_ERROR												int
#endif

//Error definitions
#ifndef	OCL_SUCCESS
#define OCL_SUCCESS												0x00000000
#endif

#define OCL_ERROR_OPEN_FILE_FAILED								0x82120001				//open file failed
#define OCL_ERROR_READ_FILE_FAILED								0x82120002
#define OCL_ERROR_WRITE_FILE_FAILED								0x81210003
#define OCL_ERROR_INVALID_ARGS									0x82120010	//-2112749552	Invalid arguments 
#define OCL_ERROR_PLATFORM_NOT_FOUND							0x82120011				//platform not found
#define OCL_ERROR_ALLOC_MEM_FAILED								0x82120012				//allocate memory failed
#define OCL_ERROR_ALIGN_PTR_FAILED								0x82120013
#define OCL_ERROR_ALIGN_INT_FAILED								0x82120014
#define OCL_ERROR_FILE_FORMAT_INVALID							0x82120015
#define OCL_ERROR_INVALID_SIZE									0x82120016
#define OCL_ERROR_INVALID_VALUE									0x82120017
#define OCL_ERROR_INVALID_DEVICE								0x82120018
#define OCL_ERROR_NULL_PTR										0x82120019

//Platform name definitions
#define OCL_PLATFORM_NAME_INTEL									"Intel(R) OpenCL"
#define OCL_PLATFORM_NAME_NVIDIA								""
#define OCL_PLATFORM_NAME_AMD									""

//define check error macros
#define oclCheckErrorAndExit(a, b, funcName, pCleanUp)					__oclCheckError(a, b, funcName, OCL_TRUE, pCleanUp, __FILE__, __LINE__)
//#define oclCheckErrorAndEixt(a, b, funcName)							oclCheckErrorAndExitEX(a, b, funcName, NULL)
#define oclCheckError(a, b, funcName)									__oclCheckError(a, b, funcName, OCL_FALSE, NULL, __FILE__, __LINE__)
#define oclCheckErrorAndReturn(a, b, funcName, retValue, pCleanUp)		if(__oclCheckError(a, b, funcName, OCL_FALSE, pCleanUp, __FILE__, __LINE__)==CL_FALSE) return retValue
//Get platform macro
#define oclGetIntelPlatform										oclGetPlatformByName(OCL_PLATFORM_NAME_INTEL)
#define oclGetAMDPlatform										oclGetPlatformByName(OCL_PLATFORM_NAME_AMD)
#define oclGetNvidiaPlatform									oclGetPlatformByName(OCL_PLATFORM_NAME_NVIDIA)


#define oclLoadKernelSource(path, __kernelSourceString)			oclLoadKernelSourceEX(path, __kernelSourceString, NULL)	

#define OCL_ALIGN_BMP_ROW										4		//Default image row align in byte
#define OCL_ALIGN_HOST_MEM_DEFAULT								32		//default allocated host memory align
//Define Image type
#define OCL_IMAGE_TYPE_BMP										
//
// int oclRound(double a)
//{
//#ifdef __SSE2__
//	__m128d d = _mm_load_sd(&a);
//	return _mm_cvtsd_si32(d);
//#else
//	
//#endif
//
//}
//

//internal align failed macros
#define IOCL_ALIGN_FAILED(errorCode, funcName)			ioclAlignFailed(errorCode, funcName, __FILE__, __LINE__)
#define IOCL_ALIGN_PTR_FAILED							IOCL_ALIGN_FAILED(OCL_ERROR_ALIGN_PTR_FAILED, "oclAlignPtr")
#define IOCL_ALIGN_INT_FAILED							(int)IOCL_ALIGN_FAILED(OCL_ERROR_ALIGN_INT_FAILED, "oclAlign")

//Align pointer macro if success return aligned ptr otherwise return 0 and printf error msg
#define oclAlignPtr(ptr, align)																							\
	(((align)&(align-1))==0?((void*)(((size_t)ptr + align - 1)&(-align))):(IOCL_ALIGN_PTR_FAILED))
//Align int macro if success return aligned int otherwise return 0 and printf error msg
#define oclAlign(n, align)																								\
	(((((align)&((align)-1))==0)&&((n)<=OCL_INT_MAX))?(((n)+(align)-1)&(-(align))):IOCL_ALIGN_INT_FAILED)

#define oclAlignSize(size, align)						oclAlign(size, align)

#define oclFree(ptr)									free(ptr); ptr=NULL

//internal used only
OCL_INTERNAL void* ioclAlignFailed(OCL_ERROR errorCode, const char* funcName, const char* file, const int line)
{
	printf("ERROR: %i from %s in file %s at line %i\n\n", errorCode, funcName, file, line);
	return (void*)0;
	//exit(errorCode);
}
//check error, internal use only
//check @sample==@ref if True the function return OCL_TRUE and do nothing,
//otherwise return OCL_FALSE and print errors, exit and clean up if necessary.
OCL_INTERNAL OCL_BOOL __oclCheckError(cl_int sample, cl_int ref, const char* funcName, int isExit, void (*pcleanup)(int), const char* file, const int line)
{
	if((sample)!=(ref))
	{
#ifdef __DEBUG__
		printf("have an error #%d from %s at line %d, in file %s\n", sample, funcName, line, file);
#endif
		
		if(pcleanup != NULL)
		{
			pcleanup((sample)!=0?(sample):-1);
			if(isExit)
			{
#ifdef __DEBUG__
				printf("Exiting.....\n");
#endif
				exit((sample)!=0?(sample):-1);
			}
		}
		return OCL_FALSE;
	}
	return OCL_TRUE;
}

//intialize data and all have the same value
OCL_INLINE OCL_ERROR oclQuickInitData(char** data, size_t dataSize, char value)
{
	OCL_ERROR error=OCL_SUCCESS;
	const char funcName[] = "oclQickInitData";
	if(!oclCheckError((data!=NULL)&&(!OCL_CHECK_OUT_OF_RANGE(dataSize, OCL_DATA_MIN_SIZE, OCL_DATA_MAX_SIZE)), OCL_TRUE, funcName))
	{
		return OCL_ERROR_INVALID_ARGS;
	}
	if(*data==NULL)
		*data=(char*)malloc(dataSize);
	if(!oclCheckError(*data!=NULL, OCL_TRUE, funcName))
		return OCL_ERROR_ALLOC_MEM_FAILED;

	memset(*data, value, dataSize);
	return OCL_SUCCESS;
}
//random initialize data

OCL_EXTERN_C int oclRoundUpGlobalSize(int group_size, int global_size);
OCL_EXTERN_C cl_uint oclGetNumComputeUnits(cl_device_id deviceID, cl_int* error);
OCL_EXTERN_C cl_ulong oclGetLocalMemSize(cl_device_id deviceID, cl_int* error);
OCL_EXTERN_C size_t oclGetMaxWorkGroupSize(cl_device_id deviceID, cl_int* error);
OCL_EXTERN_C cl_uint oclGetMaxWorkItemDememsions(cl_device_id deviceID, cl_int* error);
OCL_EXTERN_C cl_int oclGetMaxWorkItemSizes(cl_device_id deviceID, cl_uint numDememsions, size_t* size);

//Get Device Execution Capabilites
OCL_EXTERN_C cl_device_exec_capabilities oclGetDevExecCap(cl_device_id deviceID, cl_int* error);

//Get Kernel source path from keyboard
OCL_EXTERN_C void oclGetKernelSourcePath(char* path, int num_chars);

//Load kernel sorce file
OCL_EXTERN_C int oclLoadKernelSourceEX(const char* path, char** __kernelSourceString, long* kernelSourceSize);

//auto fill float array @arr
OCL_EXTERN_C void oclFillFloatArray(float* arr, int size);

//Display Platform information
OCL_EXTERN_C cl_int oclDisplayPlatformInfo(cl_platform_id platformID, cl_platform_info platformInfo);

//get a number form keyboard
OCL_EXTERN_C int oclGetNumFromKeyboard();

//check error
//check @sample==@ref if True the function return OCL_TRUE and do nothing,
//otherwise return OCL_FALSE and print errors, exit and clean up if necessary.

//Get selected platform from keyboard.
OCL_EXTERN_C cl_int oclGetSelectedPlatform();

//display device information
OCL_EXTERN_C cl_int oclDisplayDeviceInfo(cl_device_id deviceID, cl_device_info deviceInfo);

//Get selected device from kerboard.
OCL_EXTERN_C cl_int oclGetSelectedDevice();

//Display System information
OCL_EXTERN_C cl_int oclDisplaySystemInfo();

//Get platform by platform name
OCL_EXTERN_C cl_platform_id oclGetPlatformByName(const char* platformName, cl_int* error);

//Get build fail log
OCL_EXTERN_C cl_int oclDisplayBuildFailLog(cl_program program, cl_device_id deviceID);

//get build options from keyboard
OCL_EXTERN_C int oclGetBuildOptions(char* buildOpts);
//save image to BMP format(windows only)
#ifdef _WIN32
OCL_EXTERN_C OCL_ERROR oclSaveDataAsBMP(const char* data, size_t pixelSize, int width, int height, const char* fileName);
#endif

//save image to PPM file
//@data handle to buffer of data
//@numChanels Number of chanels of the image that has one or three chanels, each chanel has size of one byte.
OCL_EXTERN_C OCL_ERROR oclSaveDataAsPPM(const char* data, int numChanels, int width, int height, const char* fileName);

//Load image from PPM file
//
OCL_EXTERN_C OCL_ERROR oclLoadDataFromPPM(char** data, int* numChanels, int* width, int* height, const char* fileName);


//*********************************************************************************************************************
//compare two float arrays (use L2-Norm) with an eps tolerence for equality
//return true if @ref and @data are identical, otherwise return false
//*********************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareL2NormEpsTolerance(const float* ref, const float* data, 
													const unsigned int len, const float eps);

//*******************************************************************************************************************
//compare two float arrays
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareFloat(const float* ref, const float* data, const unsigned int lengh);

//*******************************************************************************************************************
//compare two integer arrays
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareInt(const int* ref, const int* data, const unsigned int lengh);
//*******************************************************************************************************************
//compare two unsigned integer arrays, with epsilon and threshold
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareUint(const unsigned int* ref, const unsigned int* data, const unsigned int lengh, 
									const float eps, const float thresold);
//*******************************************************************************************************************
//compare two unsigned char arrays
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C	OCL_BOOL oclCompareUchar(const unsigned char* ref, const unsigned char* data, const unsigned int lengh);

//*******************************************************************************************************************
//compare two unsigned char arrays (include Threshold for number of pixel we can have errors)
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareUcharThres(const unsigned char* ref, const unsigned char* data, const unsigned int lengh,
										const float eps, const float thres);
//*******************************************************************************************************************
//compare two integer arrays with esp tolerance for equality
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareUcharEps(const unsigned char* ref, const unsigned char* data, const unsigned int lengh,
										const float eps);
//*******************************************************************************************************************
//compare two float arrays with esp tolerence for equality
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareFloatEps(const float* ref, const float* data, const unsigned int lengh, const float eps);
//*******************************************************************************************************************
//compare two float arrays with an eps tolerence for equality and a Threshold for number of pixel errors
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareFloatEpsThres(const float* ref, const float* data, const unsigned int lengh,
											const float eps, const float thres);
#endif