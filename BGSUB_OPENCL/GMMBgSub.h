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



#ifndef __GMM_BG_SUB1__
#define __GMM_BG_SUB1__

#include "__predefine.h"
#ifndef __OPENCL__
#include "__GMMBgSub.h"
#endif


#include <stdlib.h>
#include <memory.h>
#include <assert.h>

#ifdef __OPENCL__
#include <CL\cl.h>
#else
#endif

#ifdef __USE_OPENCV__
//#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#else
#endif

#ifdef  __USE_OPENCV__
#define IMAGE						IplImage
#define GET_IMAGE_DATA(imgPtr)		imgPtr->imageData
#else

#endif

#ifdef __OPENCL__
#define	 DEVICE_BUFFER		cl_mem
#else
#endif
#ifdef __PROC_MULTIFRAME__
#define GMM_NUM_FRAMES_PER_LOAD						2
#else
#define GMM_NUM_FRAMES_PER_LOAD						1
#endif
#define GMM_DEVICE_INPUT_IMAGE_NUM_CHANELS			4
#define GMM_HOST_INPUT_IMAGE_NUM_CHANELS			3
#define GMM_DEVICE_OUTPUT_IMAGE_NUM_CHANELS			1
#define GMM_HOST_OUTPUT_IMAGE_NUM_CHANELS			1
#define GMM_IMAGE_DEPTH_IN_BYTES					1
#define GMM_IMAGE_DEPTH								8
#define GMM_DEVICE_INPUT_PIXEL_SIZE					(GMM_DEVICE_INPUT_IMAGE_NUM_CHANELS*GMM_IMAGE_DEPTH_IN_BYTES)
#define GMM_DEVICE_OUTPUT_PIXEL_SIZE				(GMM_DEVICE_OUTPUT_IMAGE_NUM_CHANELS*GMM_IMAGE_DEPTH_IN_BYTES)
#define GMM_HOST_INPUT_PIXEL_SIZE					GMM_HOST_INPUT_IMAGE_NUM_CHANELS*GMM_IMAGE_DEPTH_IN_BYTES
#define GMM_HOST_OUTPUT_PIXEL_SIZE					GMM_HOST_OUTPUT_IMAGE_NUM_CHANELS*GMM_IMAGE_DEPTH_IN_BYTES

//Define GMM Info
#define GMM_INFO_ERROR								-1
#define GMM_INFO_NUM_FRAMES_PER_LOAD				100
#define GMM_INFO_FRAME_WIDTH						200
#define GMM_INFO_FRAME_HEIGHT						300
#define GMM_INFO_FRAME_SIZE							400
#define GMM_INFO_FRAME_DEPTH						500
#define GMM_INFO_OUTPUT_FRAME_NC					600
#define GMM_INFO_INPUT_FRAME_NC						700

//
#define gmmGetNumFrames()							gmmGetInfo(GMM_INFO_NUM_FRAMES_PER_LOAD)
#define gmmGetFrameWidth()							gmmGetInfo(GMM_INFO_FRAME_WIDTH)
#define gmmGetFrameHeight()							gmmGetInfo(GMM_INFO_FRAME_HEIGHT)
#define gmmGetFrameSize()							gmmGetInfo(GMM_INFO_FRAME_SIZE)
#define gmmGetFrameDepth()							gmmGetInfo(GMM_INFO_FRAME_DEPTH)
#define gmmGetInputFrameNumChannels()				gmmGetInfo(GMM_INFO_INPUT_FRAME_NC)
#define gmmGetOutpuFrameNumChannels()				gmmGetInfo(GMM_INFO_OUTPUT_FRAME_NC)


//define constant parameters
#define GMM_CONST_PARAMS_MAX_NUM_GAUSS			4
#define GMM_CONST_PARAMS_ALPHA					0.002f
#define GMM_CONST_PARAMS_SIGMA0					15.0f
#define GMM_CONST_PARAMS_Cthr					16.0f
#define GMM_CONST_PARAMS_CLOSE_THR				9.0f
#define GMM_CONST_PARAMS_ONE_MINUS_CF			0.9f
#define GMM_CONST_PARAMS_ALPHA_MUL_CT			0.0001f	//alpha*Ct
#define GMM_CONST_PARAMS_TAU					0.5f

#define GMM_CONST_PARAMS_SHADOW_VALUE			127
#define GMM_CONST_PARAMS_BG_VALUE				0
#define GMM_CONST_PARAMS_FG_VALUE				255

typedef struct GMMBgSub
{
	IMAGE**		h_inputFrames;
	IMAGE**		h_outputFrames;
	int   nImgs;
	int imgWidth;
	int imgHeight;
	int frameSize; //width*height
	//int h_inputFrameSize_inBytes;	//widthStep*height, aligned to 16 times the size of char4
	//int h_inputFrameSize_inPixels; //width*height
	//int h_outputFrameSize_inBytes;// aligned to 16 times the size of char
	//int h_outputFrameSize_inPixels;
	char* input_pinnedPtr;
	char* output_pinnedPtr;

#if (defined __OPENCL__)//&&(!defined __OPENCL_CPU__)
	DEVICE_BUFFER	input_pinned_buffer;
	size_t			input_pinned_size;//in bytes 
	size_t			input_pinned_offset;
	DEVICE_BUFFER	output_pinned_buffer;
	size_t			output_pinned_size;//in bytes 
	size_t			output_pinned_offset;

	DEVICE_BUFFER	input_buffer;
	cl_int			inputBufferSize;
	DEVICE_BUFFER	output_buffer;
	cl_int			outputBufferSize;

	size_t			input_mem_offset;
	size_t			output_mem_offset;
#if __OVERLAP_COPY__==2
	size_t			inputHalfBuffer[2];
	size_t			outputHalfBuffer[2];
#endif
#if __OVERLAP_COPY__==1
	size_t			inputHalfBuffer;
	size_t			outputHalfBuffer;
#endif
	//GMM parameters
	cl_mem gmmMuSig;
	cl_mem gmmWeight;
	cl_mem gmmNumGauss;
	cl_int	isDetectShadow;
#if __OVERLAP_COPY__ ==1
	//
	//the copy command queue
	cl_command_queue	copyCmdQueue;
	//the execution command queue
	cl_command_queue	execCmdQueue;
#endif

#if __OVERLAP_COPY__ == 2
	cl_command_queue cmdQueue1;
	cl_command_queue cmdQueue2;
#endif

	size_t*	global_work_size;
	size_t*	local_work_size;
	//int		isAutoSelectWorkSizes;
	///
	cl_platform_id	platformID;
	cl_device_id	deviceID;
	//int				deviceType;
	cl_context		context;
	char*			gmmProgramSource;
	cl_program		gmmProgram;
#if __OVERLAP_COPY__ == 1
	cl_kernel		gmmKernel;
#endif
#if __OVERLAP_COPY__==2
	cl_kernel		gmmKernel[2];
#endif
	//char			gmmKernelName[256];
	char			buildOpts[1024];
	//int				isProfiling;
#ifdef __GPU_TIMER__
	cl_ulong		startTime, endTime;
	float			executionTimeInMilliseconds;
	int				frameCounter;
	cl_event		timingEvent;
#endif
#endif

}GMMBgSub;

//

//

__API int gmmInitGMMBgSub(int width, int height/*, int nImages*/);

__API int releaseGMMBgSub();

#if __OVERLAP_COPY__==1
__API int gmmUpdateGMMBgSub(CvCapture* capture);
#endif

#if __OVERLAP_COPY__==2
__API int gmmUpdateGMMBgSub();
#endif

__API int gmmGetOutputFrames(IplImage** outputFrames, int nFrames);

__API int gmmPutInputFrames1(IplImage** inputFrames, int nFrames);

__API int gmmGetInfo(int gmmInfo);

#ifdef __GPU_TIMER__
__API float gmmGetKernelExecTimeInMilliseconds();
__API int gmmGetProcessedFrames();
#endif

#endif
