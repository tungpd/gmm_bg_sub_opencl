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



#include "GMMBgSub.h"
#include <malloc.h>
#include <stdlib.h>
#include <oclUtils.h>


static GMMBgSub* gmm=NULL;
//predefine
static int gmmPutInputFrames(CvCapture* capture);

void cleanUp(int iExitCode)
{
	if(gmm)
	{
		if(gmm->context)
			clReleaseContext(gmm->context);
#if __OVERLAP_COPY__<=1
		if(gmm->copyCmdQueue)
			clReleaseCommandQueue(gmm->copyCmdQueue);
		if(gmm->execCmdQueue)
			clReleaseCommandQueue(gmm->execCmdQueue);
		if(gmm->gmmKernel)
			clReleaseKernel(gmm->gmmKernel);
		if(gmm->output_buffer)
			clReleaseMemObject(gmm->output_buffer);
		if(gmm->output_buffer)
			clReleaseMemObject(gmm->output_buffer);
		if(gmm->input_buffer)
			clReleaseMemObject(gmm->input_buffer);
		if(gmm->input_buffer)
			clReleaseMemObject(gmm->input_buffer);
#endif
#if __OVERLAP_COPY__>=2
		if(gmm->cmdQueue1)
			clReleaseCommandQueue(gmm->cmdQueue1);
		if(gmm->cmdQueue2)
			clReleaseCommandQueue(gmm->cmdQueue2);
		if(gmm->gmmKernel[0])
			clReleaseKernel(gmm->gmmKernel[0]);
		if(gmm->gmmKernel[1])
			clReleaseKernel(gmm->gmmKernel[1]);
		if(gmm->input_buffer)
			clReleaseMemObject(gmm->input_buffer);
		if(gmm->output_buffer)
			clReleaseMemObject(gmm->output_buffer);
#endif
		if(gmm->gmmProgram)
			clReleaseProgram(gmm->gmmProgram);
		if(gmm->input_pinned_buffer)
			clReleaseMemObject(gmm->input_pinned_buffer);
		if(gmm->output_pinned_buffer)
			clReleaseMemObject(gmm->output_pinned_buffer);
		if(gmm->gmmMuSig)
			clReleaseMemObject(gmm->gmmMuSig);
		if(gmm->gmmWeight)
			clReleaseMemObject(gmm->gmmWeight);
		if(gmm->gmmNumGauss)
			clReleaseMemObject(gmm->gmmNumGauss);
		if(gmm->gmmProgramSource)
			free(gmm->gmmProgramSource);

		if(gmm->h_inputFrames)
		{
			for(int i = 0; i < gmm->nImgs; i++)
			{
				if(gmm->h_inputFrames[i])
					cvReleaseImage(&(gmm->h_inputFrames[i]));
			}
		}
		if(gmm->h_outputFrames)
		{
			for(int i = 0; i < gmm->nImgs; i++)
			{
				if(gmm->h_outputFrames[i])
					cvReleaseImage(&(gmm->h_outputFrames[i]));
			}
		}
		if(gmm->local_work_size)
			free(gmm->local_work_size);
		if(gmm->global_work_size)
			free(gmm->global_work_size);

		oclFree(gmm);
		gmm=NULL;
	}
#ifdef __DEBUG__
	cvDestroyWindow("input video");
#endif
	exit(iExitCode);
}
void (*pCleanUp)(int)=&cleanUp;

static int iWriteToInputPinned(IplImage* inputFrame, int frameCount)
{
	const char* funcname="iWriteToInputPinned";
	int status=OCL_SUCCESS;
	oclCheckErrorAndReturn(((inputFrame!=NULL)&&(inputFrame->nChannels==3)&&(inputFrame->width==gmm->imgWidth)&&(inputFrame->imageData!=NULL)&&
		(inputFrame->height==gmm->imgHeight)&&(inputFrame->depth==GMM_IMAGE_DEPTH)), OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);

	char* imageData=inputFrame->imageData;
	char* pinnedBuffer=gmm->input_pinnedPtr + frameCount*gmm->input_pinned_offset;
	for(int row=0; row<gmm->imgHeight; row++)
	{
		for(int col=0; col<gmm->imgWidth; col++)
		{
			memcpy(pinnedBuffer, imageData, 4/*GMM_HOST_INPUT_PIXEL_SIZE*/);
			pinnedBuffer+=GMM_DEVICE_INPUT_PIXEL_SIZE;
			imageData+=GMM_HOST_INPUT_PIXEL_SIZE;
		}
		imageData+=(inputFrame->widthStep-(GMM_HOST_INPUT_PIXEL_SIZE*gmm->imgWidth));
		//pinnedBuffer+=((gmm->imgWidth<<2)-(GMM_DEVICE_INPUT_PIXEL_SIZE*gmm->imgWidth));
	}
	return status;
}
static int iReadFromOutputPinned(IplImage* outputFrame, int frameCount)
{
	const char* funcname="iReadFromOutputPinned";
	int status=OCL_SUCCESS;
	oclCheckErrorAndReturn(((outputFrame!=NULL)&&(outputFrame->nChannels==1)&&(outputFrame->width==gmm->imgWidth)&&
		(outputFrame->height==gmm->imgHeight)&&(outputFrame->depth==GMM_IMAGE_DEPTH)&&(outputFrame->imageData!=NULL)), OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);

	char* imageData=outputFrame->imageData;
	char* pinnedBuffer=gmm->output_pinnedPtr + frameCount*gmm->output_pinned_offset;
	int pinnedWidthStep=gmm->imgWidth*GMM_DEVICE_OUTPUT_PIXEL_SIZE;

	for(int row=0; row<gmm->imgHeight; row++)
	{
		memcpy(imageData, pinnedBuffer, pinnedWidthStep);
		imageData+=outputFrame->widthStep;
		pinnedBuffer+=pinnedWidthStep;
	}

	return status;
}
static int gmmInitSystem()
{
	const char* funcName="initSystem";
	printf("%s", funcName);
	if(!gmm)
		gmm=(GMMBgSub*)malloc(sizeof(GMMBgSub));
	oclCheckErrorAndReturn(gmm!=NULL, OCL_TRUE, funcName, OCL_ERROR_ALLOC_MEM_FAILED, pCleanUp);
	memset(gmm, 0, sizeof(GMMBgSub));

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//display platforms
	oclDisplaySystemInfo();

	int status=0;
	cl_platform_id* platformIDs;
	cl_device_id*	cdDevices;
	cl_uint selectedPlatform=0;
	cl_uint numDevices=0;
	cl_uint numPlatforms;
	cl_uint selectedDevice=0;
	cl_uint numComputeUnits=0;

	//get platforms
	status=clGetPlatformIDs(0, 0, &numPlatforms);
	platformIDs=(cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
	status|=clGetPlatformIDs(numPlatforms, platformIDs, 0);
	oclCheckError(status, CL_SUCCESS, funcName);

	//select one platform
	do{
		selectedPlatform=oclGetSelectedPlatform();
	}
	while(0 > selectedPlatform || selectedPlatform >= numPlatforms);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	gmm->platformID=platformIDs[selectedPlatform];
	free(platformIDs);
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	//display devices of the selected platform
	status=clGetDeviceIDs(gmm->platformID, CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
	oclCheckError(status, CL_SUCCESS, funcName);
	cdDevices=(cl_device_id*) malloc(numDevices*sizeof(*cdDevices));
	status=clGetDeviceIDs(gmm->platformID, CL_DEVICE_TYPE_ALL, numDevices, cdDevices, 0);
	oclCheckError(status, CL_SUCCESS, funcName);
	printf("all devices of the selected platform: \n");
	
	for(cl_uint j=0; j<numDevices; j++)
	{
		printf("Device #%d\n", j);
		printf("**************************************\n");
		printf("Device name:	\n");
		oclDisplayDeviceInfo(cdDevices[j], CL_DEVICE_NAME);
		printf("Device Vendor:\n");
		oclDisplayDeviceInfo(cdDevices[j], CL_DEVICE_VENDOR);
		printf("Max compute units: \n");
		oclDisplayDeviceInfo(cdDevices[j], CL_DEVICE_MAX_COMPUTE_UNITS);
	}
	//select one device
	selectedDevice=oclGetSelectedDevice();
	while(selectedDevice<0 || selectedDevice>=numDevices)
	{
		printf("please enter a number from 0 to %d: \n", numDevices-1);
		selectedDevice=oclGetSelectedDevice();
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	gmm->deviceID=cdDevices[selectedDevice];

	free(cdDevices);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//get number of compute units of the device
	numComputeUnits=oclGetNumComputeUnits(gmm->deviceID, &status);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcName, status, pCleanUp);
	printf("Number of compute uints of the device is:	%d\n", numComputeUnits);
	//get maximum number of threads per work group
	size_t maxNumThreadsPerGroup;
	status=clGetDeviceInfo(gmm->deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxNumThreadsPerGroup), &maxNumThreadsPerGroup, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcName, status, pCleanUp);
	printf("\n\nMaximum number of threads per work group:	%d\n", maxNumThreadsPerGroup);
	//
	cl_uint maxDim;
	status=clGetDeviceInfo(gmm->deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxDim), &maxDim, NULL);
	size_t* maxNumThreadsPerComputeUnits=(size_t*)malloc(sizeof(size_t)*maxDim);
	status=clGetDeviceInfo(gmm->deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxDim*sizeof(size_t), (void*)maxNumThreadsPerComputeUnits, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcName, status, pCleanUp);
	size_t maxGlobalSize=1;
	for(unsigned int i=0; i<maxDim; i++)
	{
		printf("maxNumThreadsPerComputeUnits[%d]=%d\n", i, maxNumThreadsPerComputeUnits[i]);
		maxGlobalSize*=maxNumThreadsPerComputeUnits[i];
	}
	printf("\n\nMaximum number of threads per multiprocessor:	%d\n", maxGlobalSize);

	//get device type
	int deviceType=0;
	cl_device_type __deviceType;
	char deviceVendor[1024];
	status=clGetDeviceInfo(gmm->deviceID, CL_DEVICE_TYPE, sizeof(__deviceType), &__deviceType, NULL);
	status|=clGetDeviceInfo(gmm->deviceID, CL_DEVICE_VENDOR, sizeof(deviceVendor), &deviceVendor, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcName, status, pCleanUp);

	if(__deviceType==CL_DEVICE_TYPE_CPU)
	{
		if(strstr(deviceVendor, "Intel"))
			deviceType=GMM_DEVICE_TYPE_CPU_INTEL;
		if(strstr(deviceVendor, "AMD"))
			deviceType=GMM_DEVICE_TYPE_CPU_AMD;
	}
	if(__deviceType==CL_DEVICE_TYPE_GPU)
	{
		if(strstr(deviceVendor, "NVIDIA"))
			deviceType=GMM_DEVICE_TYPE_GPU_NVIDIA;
		if(strstr(deviceVendor, "AMD"))
			deviceType=GMM_DEVICE_TYPE_GPU_AMD;
	}
	oclCheckErrorAndReturn(deviceType!=0, OCL_TRUE, funcName, OCL_ERROR_INVALID_DEVICE, pCleanUp);
	printf("end select device type/n/n");
	//get max size of memory object allocation
	memsize_t memsize;
	printf("\n\nGet CL_DEVICE_MAX_MEM_ALLOC_SIZE......");
	status=clGetDeviceInfo(gmm->deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(memsize), &memsize, 0);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcName, status, pCleanUp);
	printf("\n\n\nMaximum memory allocated size:	%ul\n\n\n", memsize);

	//create the context
	printf("Creating context.......");
	gmm->context = clCreateContext(0, 1, &gmm->deviceID, 0, 0, &status);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcName, status, pCleanUp);

	
	//create two command queues: execQueue and copyQueue
#ifdef __GPU_TIMER__
	cl_command_queue_properties cmdProperties=CL_QUEUE_PROFILING_ENABLE;
#else
	cl_command_queue_properties cmdProperties=0;
#endif
#if __OVERLAP_COPY__ ==1
	gmm->execCmdQueue = clCreateCommandQueue(gmm->context, gmm->deviceID, cmdProperties, &status);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcName, status, pCleanUp);
	
	gmm->copyCmdQueue = clCreateCommandQueue(gmm->context, gmm->deviceID, cmdProperties, &status);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcName, status, pCleanUp);
#endif
#if __OVERLAP_COPY__==2
	gmm->cmdQueue1 = clCreateCommandQueue(gmm->context, gmm->deviceID, cmdProperties, &status);
	gmm->cmdQueue2 = clCreateCommandQueue(gmm->context, gmm->deviceID, cmdProperties, &status);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcName, status, pCleanUp);
#endif

	//compute global work size and local work size
	
	gmm->global_work_size=(size_t*)malloc(sizeof(size_t));
	gmm->local_work_size=(size_t*)malloc(sizeof(size_t));
	int isAutoSelectWorkSizes = 0;
	printf("\n\nDo you want set up global work size and local work size manually or automatically (Y/N)?: ");
	fflush(stdin);
	char c;
	scanf("%c", &c);
	if(c=='y'||c=='Y')
		isAutoSelectWorkSizes=1;
	else
		isAutoSelectWorkSizes=0;

	if(isAutoSelectWorkSizes)
	{
		if(deviceType==GMM_DEVICE_TYPE_CPU_INTEL||deviceType==GMM_DEVICE_TYPE_CPU_AMD)
		{
			gmm->global_work_size[0]=maxGlobalSize + 1;
			free(gmm->local_work_size);
			gmm->local_work_size=NULL;
		}
		if(deviceType==GMM_DEVICE_TYPE_GPU_AMD||deviceType==GMM_DEVICE_TYPE_GPU_NVIDIA)
		{
			

			gmm->local_work_size[0]=maxNumThreadsPerGroup/4;
			gmm->global_work_size[0]=OCL_CLAMP(numComputeUnits*(gmm->local_work_size[0]),
				GMM_NUM_WORK_GROUPS_PER_COMPUTE_UNIT*numComputeUnits*gmm->local_work_size[0],
				maxNumThreadsPerComputeUnits[0]);
		}
	}
	else
	{
		printf("\n\n\nPlease enter number of threads per work group:	");
		fflush(stdin);
		int n;
		scanf("%d", &n);
		
		gmm->local_work_size[0]=(size_t)OCL_CLAMP(64, (size_t)n, maxNumThreadsPerGroup);
		printf("\n\nPlease enter number of work groups:	");
		fflush(stdin);
		scanf("%d",&n);
		gmm->global_work_size[0]=OCL_CLAMP(numComputeUnits*gmm->local_work_size[0],
			n*gmm->local_work_size[0], maxNumThreadsPerComputeUnits[0]);
		//print work sizes
		printf("\n\nLocal work size:	%d\n", gmm->local_work_size[0]);
		printf("Number of Workgroups:	%d\n",gmm->global_work_size[0]/gmm->local_work_size[0]);
	

	}
	
	//get build options
	//
	oclGetBuildOptions(gmm->buildOpts);
	//display build opts
	printf("\nBuild options: %s\n", gmm->buildOpts);
	//Build the kernel
	//
	//get the path of the opencl kernel source file
	char path[1024];
	//set kernel name
	
	oclGetKernelSourcePath(path, 1024);
	oclCheckErrorAndReturn(path!=NULL, OCL_TRUE, funcName, OCL_ERROR_OPEN_FILE_FAILED, pCleanUp);
	gmm->gmmProgramSource=NULL;
	//load kernel source from file
	
	status=oclLoadKernelSource(path, &gmm->gmmProgramSource);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcName, status, pCleanUp);
	//create the program
	gmm->gmmProgram = clCreateProgramWithSource(gmm->context, 1, (const char**)&gmm->gmmProgramSource, NULL, &status);
	//build the program
	status=clBuildProgram(gmm->gmmProgram, 1, &gmm->deviceID, gmm->buildOpts, NULL, NULL);
	//oclCheckErrorAndReturn(status, OCL_SUCCESS, funcName, status);

	oclDisplayBuildFailLog(gmm->gmmProgram, gmm->deviceID);

	//
	//Create the kernel
	//
#if __OVERLAP_COPY__==2
	gmm->gmmKernel[0] = clCreateKernel(gmm->gmmProgram, GMM_KERNEL_NAME, &status);
	gmm->gmmKernel[1] = clCreateKernel(gmm->gmmProgram, GMM_KERNEL_NAME, &status);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcName, status, pCleanUp);
#endif

#if __OVERLAP_COPY__==1
	gmm->gmmKernel = clCreateKernel(gmm->gmmProgram, GMM_KERNEL_NAME, &status);
#endif
	return status;
	
	
}

__IMPORTS int gmmInitGMMBgSub(int width, int height/*, int nImages*/)
{
	const char* funcname = "gmmInitGMMBgSub";	//for checking errors
	//GMMBgSub* gmmRet= (GMMBgSub*)malloc(sizeof(GMMBgSub));
	int status=CL_SUCCESS;
	status=gmmInitSystem();
	oclCheckErrorAndExit(status, CL_SUCCESS, funcname, pCleanUp);
	gmm->isDetectShadow=1;

	gmm->nImgs= 2*GMM_NUM_FRAMES_PER_LOAD;
	gmm->imgHeight=height;
	gmm->imgWidth=width;
	
	gmm->frameSize=oclAlignSize(width*height, 16);//frame size in pixels

	//create input and output frames
	gmm->h_inputFrames=(IMAGE**)malloc(sizeof(IMAGE*)*gmm->nImgs);
	gmm->h_outputFrames=(IMAGE**)malloc(sizeof(IMAGE*)*gmm->nImgs);

	for(int i=0; i<gmm->nImgs; i++)
	{
		gmm->h_inputFrames[i]=cvCreateImage(cvSize(width, height), GMM_IMAGE_DEPTH, GMM_HOST_INPUT_IMAGE_NUM_CHANELS);
		gmm->h_outputFrames[i]=cvCreateImage(cvSize(width, height), GMM_IMAGE_DEPTH, GMM_HOST_OUTPUT_IMAGE_NUM_CHANELS);
		//check error
		oclCheckErrorAndExit(gmm->h_inputFrames[i]!=NULL, OCL_TRUE, funcname, pCleanUp);
		oclCheckErrorAndExit(gmm->h_outputFrames[i]!=NULL, OCL_TRUE, funcname, pCleanUp);
	}
	
	gmm->input_pinned_offset=oclAlignSize(gmm->frameSize*GMM_DEVICE_INPUT_PIXEL_SIZE, 16*GMM_DEVICE_INPUT_PIXEL_SIZE);
	gmm->output_pinned_offset=oclAlignSize(gmm->frameSize*GMM_DEVICE_OUTPUT_PIXEL_SIZE, 16*GMM_DEVICE_OUTPUT_PIXEL_SIZE);

	gmm->input_mem_offset=gmm->input_pinned_offset;
	gmm->output_mem_offset=gmm->output_pinned_offset;

	gmm->input_pinned_size=gmm->input_pinned_offset*gmm->nImgs;
	gmm->output_pinned_size=gmm->output_pinned_offset*gmm->nImgs;

	//alloc pinned memory space
	gmm->input_pinned_buffer=clCreateBuffer(gmm->context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, gmm->input_pinned_size, NULL, &status);
	gmm->output_pinned_buffer=clCreateBuffer(gmm->context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, gmm->output_pinned_size, NULL, &status);
	oclCheckErrorAndExit(status, CL_SUCCESS, funcname, pCleanUp);

	
#if __OVERLAP_COPY__==1
	gmm->input_pinnedPtr=(char*)clEnqueueMapBuffer(gmm->copyCmdQueue, gmm->input_pinned_buffer, CL_TRUE, CL_MAP_WRITE, 0,
									gmm->input_pinned_size, 0, NULL, NULL, &status);
	gmm->output_pinnedPtr=(char*)clEnqueueMapBuffer(gmm->copyCmdQueue, gmm->output_pinned_buffer, CL_TRUE, CL_MAP_READ, 0,
										gmm->output_pinned_size, 0, NULL, NULL, &status);

	oclCheckErrorAndReturn(status, CL_SUCCESS, funcname, status, pCleanUp);

	gmm->inputHalfBuffer=0;
	gmm->outputHalfBuffer=0;
#endif
#if __OVERLAP_COPY__==2
	gmm->input_pinnedPtr=(char*)clEnqueueMapBuffer(gmm->cmdQueue1, gmm->input_pinned_buffer, CL_TRUE, CL_MAP_WRITE, 0,
									gmm->input_pinned_size, 0, NULL, NULL, &status);
	gmm->output_pinnedPtr=(char*)clEnqueueMapBuffer(gmm->cmdQueue1, gmm->output_pinned_buffer, CL_TRUE, CL_MAP_READ, 0,
										gmm->output_pinned_size, 0, NULL, NULL, &status);

	oclCheckErrorAndReturn(status, CL_SUCCESS, funcname, status, pCleanUp);

	gmm->inputHalfBuffer[0]=0;
	gmm->outputHalfBuffer[0]=0;
	gmm->inputHalfBuffer[1]=gmm->inputBufferSize/2;
	gmm->outputHalfBuffer[1]=gmm->outputBufferSize/2;

#endif
	//set pinned buffers to zero
	memset(gmm->input_pinnedPtr, 0, gmm->input_pinned_size);
	memset(gmm->output_pinnedPtr, 0, gmm->output_pinned_size);
	//
	gmm->inputBufferSize=gmm->input_pinned_size;
	gmm->outputBufferSize=gmm->output_pinned_size;
	//create device memory


	gmm->input_buffer=clCreateBuffer(gmm->context, CL_MEM_READ_ONLY, gmm->inputBufferSize, NULL, &status);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcname, status, pCleanUp);
	gmm->output_buffer=clCreateBuffer(gmm->context, CL_MEM_WRITE_ONLY, gmm->outputBufferSize, NULL, &status);
	oclCheckErrorAndReturn(status, CL_SUCCESS, funcname, status, pCleanUp);

	//allocate buffer for MuSig Params
	size_t MusigSize=GMM_CONST_PARAMS_MAX_NUM_GAUSS*sizeof(cl_float4)*gmm->frameSize;
	gmm->gmmMuSig=clCreateBuffer(gmm->context, CL_MEM_READ_WRITE, MusigSize, NULL, &status);

	cl_float4* MuSigValue = (cl_float4*)malloc(MusigSize);
	memset(MuSigValue, 0, MusigSize);
#if __OVERLAP_COPY__ ==1
	clEnqueueWriteBuffer(gmm->copyCmdQueue, gmm->gmmMuSig, CL_TRUE, 0, MusigSize, MuSigValue, 0, NULL, NULL);
#endif 
#if __OVERLAP_COPY__ ==2
	clEnqueueWriteBuffer(gmm->cmdQueue1, gmm->gmmMuSig, CL_TRUE, 0, MusigSize, MuSigValue, 0, NULL, NULL);
#endif


	//allocate buffer for Weight params
	size_t weightSize=GMM_CONST_PARAMS_MAX_NUM_GAUSS*sizeof(cl_float)*gmm->frameSize;
	gmm->gmmWeight=clCreateBuffer(gmm->context, CL_MEM_READ_WRITE, weightSize, NULL, &status);
	cl_float* weightValues = (cl_float*)malloc(weightSize);
	memset(weightValues, 0, weightSize);
#if __OVERLAP_COPY__ ==1
	clEnqueueWriteBuffer(gmm->copyCmdQueue, gmm->gmmWeight, CL_TRUE, 0, weightSize, weightValues, 0, NULL, NULL);
#endif 
#if __OVERLAP_COPY__ ==2
	clEnqueueWriteBuffer(gmm->cmdQueue1, gmm->gmmWeight, CL_TRUE, 0, weightSize, weightValues, 0, NULL, NULL);
#endif

	//allocate constant buffer for nGauss params
	size_t nGaussSize=sizeof(cl_int)*gmm->frameSize;
	gmm->gmmNumGauss=clCreateBuffer(gmm->context, CL_MEM_READ_WRITE, nGaussSize, NULL, &status);
	//set all nGauss params to zero
	cl_int* zeros=(cl_int*)malloc(nGaussSize);
	memset(zeros, 0, nGaussSize);
#if __OVERLAP_COPY__ ==1
	clEnqueueWriteBuffer(gmm->copyCmdQueue, gmm->gmmNumGauss, CL_TRUE, 0, nGaussSize, zeros, 0, NULL, NULL);
#endif 
#if __OVERLAP_COPY__ ==2
		clEnqueueWriteBuffer(gmm->cmdQueue1, gmm->gmmNumGauss, CL_TRUE, 0, nGaussSize, zeros, 0, NULL, NULL);
#endif

	//set detect shadow
	
	

	//set kernel args
	int i=0;
#if __OVERLAP_COPY__==1
#ifdef __PROC_MULTIFRAMES__
	i=0;
	cl_int hNumImages=gmm->nImgs/2;

	status=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->input_buffer);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->output_buffer);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&hNumImages);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->input_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->output_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->frameSize);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->inputHalfBuffer);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->outputHalfBuffer);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->gmmMuSig);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->gmmWeight);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->gmmNumGauss);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->isDetectShadow);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
#else
	i=0;
	cl_int hNumImages=gmm->nImgs/2;

	status=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->input_buffer);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->output_buffer);
	/*status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&hNumImages);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->input_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->output_mem_offset);*/
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->frameSize);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->inputHalfBuffer);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->outputHalfBuffer);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->gmmMuSig);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->gmmWeight);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_mem), (void*)&gmm->gmmNumGauss);
	status|=clSetKernelArg(gmm->gmmKernel, i++, sizeof(cl_int), (void*)&gmm->isDetectShadow);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
#endif
#endif
#if __OVERLAP_COPY__==2
#ifdef __PROC_MULTIFRAMES__
	//set args for kernel[0]
	i=0;
	cl_int hNumImages=gmm->nImgs/2;
	status=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->input_buffer);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->output_buffer);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&hNumImages);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->input_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->output_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->frameSize);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->inputHalfBuffer[0]);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->outputHalfBuffer[0]);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->gmmMuSig);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->gmmWeight);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->gmmNumGauss);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->isDetectShadow);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	//set args for kernel[1]
	i=0;
	status=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->input_buffer);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->output_buffer);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&hNumImages);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->input_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->output_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->frameSize);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->inputHalfBuffer[1]);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->outputHalfBuffer[1]);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->gmmMuSig);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->gmmWeight);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->gmmNumGauss);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->isDetectShadow);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
#else
	//set args for kernel[0]
	i=0;
	cl_int hNumImages=gmm->nImgs/2;
	status=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->input_buffer);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->output_buffer);
	/*status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&hNumImages);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->input_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->output_mem_offset);*/
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->frameSize);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->inputHalfBuffer[0]);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->outputHalfBuffer[0]);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->gmmMuSig);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->gmmWeight);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_mem), (void*)&gmm->gmmNumGauss);
	status|=clSetKernelArg(gmm->gmmKernel[0], i++, sizeof(cl_int), (void*)&gmm->isDetectShadow);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	//set args for kernel[1]
	i=0;
	status=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->input_buffer);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->output_buffer);
	/*status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&hNumImages);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->input_mem_offset);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->output_mem_offset);*/
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->frameSize);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->inputHalfBuffer[1]);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->outputHalfBuffer[1]);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->gmmMuSig);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->gmmWeight);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_mem), (void*)&gmm->gmmNumGauss);
	status|=clSetKernelArg(gmm->gmmKernel[1], i++, sizeof(cl_int), (void*)&gmm->isDetectShadow);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
#endif
#endif

#ifdef __GPU_TIMER__
	gmm->executionTimeInMilliseconds=0;
	gmm->frameCounter=0;
	//warming up GPU
	status=clEnqueueNDRangeKernel(gmm->execCmdQueue, gmm->gmmKernel, 1, NULL, gmm->global_work_size, gmm->local_work_size, 0, NULL, &gmm->timingEvent);
	//clFinish(gmm->execCmdQueue);
#endif

	return status;
}



//update GMM
#if __OVERLAP_COPY__==2	//not debuged
__IMPORTS int gmmUpdateGMMBgSub()
{
	const char* funcname="updateGMMBgSub";
	int status=OCL_SUCCESS;
	oclCheckErrorAndReturn((gmm!=NULL), OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);

	status=clEnqueueWriteBuffer(gmm->cmdQueue1, gmm->input_buffer, CL_FALSE, 0, gmm->inputHalfBuffer[1], (void*)gmm->input_pinnedPtr, 0, NULL, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	clFlush(gmm->cmdQueue1);
	//Lauch kernel computation, queue 0
	status=clEnqueueNDRangeKernel(gmm->cmdQueue1, gmm->gmmKernel[0], 1, NULL, gmm->global_work_size, gmm->local_work_size, 0, NULL, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	//noneblocking write of 2nd half of input data, cmdqueue2
	status=clEnqueueWriteBuffer(gmm->cmdQueue2, gmm->input_buffer, CL_FALSE, gmm->inputHalfBuffer[1], gmm->inputHalfBuffer[1], 
								(void*)(gmm->input_pinnedPtr + gmm->inputHalfBuffer[1]), 0, NULL, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	clFlush(gmm->cmdQueue1);
	clFlush(gmm->cmdQueue2);
	//Launch kernel computation, cmdQueue 1
	status=clEnqueueNDRangeKernel(gmm->cmdQueue2, gmm->gmmKernel[1], 1, NULL, gmm->global_work_size, gmm->local_work_size, 0, NULL, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	//nonblocking read of 1st half of output data, queue0
	status=clEnqueueReadBuffer(gmm->cmdQueue1, gmm->output_buffer, CL_FALSE, 0, gmm->outputHalfBuffer[1], 
										(void*)gmm->output_pinnedPtr, 0, NULL, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	clFlush(gmm->cmdQueue1);
	clFlush(gmm->cmdQueue2);
	//nonblocking read of 2nd half of output data, queue1
	status=clEnqueueReadBuffer(gmm->cmdQueue2, gmm->output_buffer, CL_FALSE, gmm->outputHalfBuffer[1], gmm->outputHalfBuffer[1],
											(void*)(gmm->output_pinnedPtr + gmm->outputHalfBuffer[1]), 0, NULL, NULL);

	clFinish(gmm->cmdQueue1);
	clFinish(gmm->cmdQueue2);

	return status;
}
#endif

#if __OVERLAP_COPY__==1
__IMPORTS int gmmUpdateGMMBgSub(CvCapture* capture)
{
	const char* funcname="updateGMMBgSub";
	int status=OCL_SUCCESS;
	oclCheckErrorAndReturn((gmm!=NULL), OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);

	

	//wait for the process of ith frame and the copy of the i+1 th intput frame to device memory complete,
	//then we swap input_buffer1 and input_buffer1, swap output_buffer1 and output_buffer2
	//then we run the kernel to process the i+1 th frame.

	clFinish(gmm->execCmdQueue);
	clFinish(gmm->copyCmdQueue);

#ifdef __GPU_TIMER__
	gmm->frameCounter+=gmm->nImgs/2; 
	status=clGetEventProfilingInfo(gmm->timingEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gmm->startTime, NULL);
	status|=clGetEventProfilingInfo(gmm->timingEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gmm->endTime, NULL);
	oclCheckError(status, OCL_SUCCESS, funcname);
	gmm->executionTimeInMilliseconds+=(gmm->endTime-gmm->startTime)*1.0e-6f;

#endif
	gmm->inputHalfBuffer=(gmm->inputHalfBuffer==0)?(gmm->inputBufferSize/2):0;
	gmm->outputHalfBuffer=(gmm->outputHalfBuffer==0)?(gmm->outputBufferSize/2):0;

	//reset the args
#ifdef __PROC_MULTIFRAMES__
	status|=clSetKernelArg(gmm->gmmKernel, 6, sizeof(cl_int), (void*)&gmm->inputHalfBuffer);
	status|=clSetKernelArg(gmm->gmmKernel, 7, sizeof(cl_int), (void*)&gmm->outputHalfBuffer);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
#else
	status|=clSetKernelArg(gmm->gmmKernel, 3, sizeof(cl_int), (void*)&gmm->inputHalfBuffer);
	status|=clSetKernelArg(gmm->gmmKernel, 4, sizeof(cl_int), (void*)&gmm->outputHalfBuffer);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
#endif
	//run the kernel
#ifdef __GPU_TIMER__
	//clFinish(gmm->execCmdQueue);
	status=clEnqueueNDRangeKernel(gmm->execCmdQueue, gmm->gmmKernel, 1, NULL, gmm->global_work_size, gmm->local_work_size, 0, NULL, &gmm->timingEvent);
#else
	status=clEnqueueNDRangeKernel(gmm->execCmdQueue, gmm->gmmKernel, 1, NULL, gmm->global_work_size, gmm->local_work_size, 0, NULL, NULL);
#endif
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	//now, on the Device, the i+1th frame are being processed........
	//
	//while the i+1th frame are being processed, we copy the ith output frame from Device memory to pinned memory,
	//and the i+2th input frame from host to pinned.
	
	status=clEnqueueReadBuffer(gmm->copyCmdQueue, gmm->output_buffer, CL_FALSE, (gmm->outputHalfBuffer==0)?(gmm->outputBufferSize/2):0, gmm->outputBufferSize/2,
		(void*)(gmm->output_pinnedPtr + ((gmm->outputHalfBuffer==0)?(gmm->outputBufferSize/2):0)), 0, NULL, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	//copy i+2th input frame from host to pinned
	
	gmmPutInputFrames(capture);

	//we wait for finishing copy the ith output frame from device to pinned
	clFinish(gmm->copyCmdQueue);

	//when copy the ith output frame from device to pinned and copy i+2th input frame from host to pinned finish,
	//then we copy i+2th input frame to device memory and ith output frame to host asynchronously.
	status=clEnqueueWriteBuffer(gmm->copyCmdQueue, gmm->input_buffer, CL_FALSE, (gmm->inputHalfBuffer==0)?(gmm->inputBufferSize/2):0, gmm->inputBufferSize/2,
		(void*)(gmm->input_pinnedPtr + ((gmm->inputHalfBuffer==0)?(gmm->inputBufferSize/2):0)), 0, NULL, NULL);
	oclCheckErrorAndReturn(status, OCL_SUCCESS, funcname, status, pCleanUp);
	//while the device copys the i+1 th frame to device memory,
	//we copy the i-2 th output frame from pinned memory to host.
	/*for(int i=0; i<gmm->nImgs; i++)
	{
		iReadFromOutputPinned(gmm->h_outputFrames[i], i);
	}
*/

	return status;
}
#endif

__IMPORTS int gmmReleaseGMMBgSub()
{
	cleanUp(0);
	return 0;
}

__IMPORTS int gmmGetOutputFrames(IplImage** outputFrames, int nFrames)
{
	const char* funcname="gmmGetOutputFrames";
	int status=OCL_SUCCESS;
#if __OVERLAP_COPY__==1
	oclCheckErrorAndReturn((gmm!=NULL)&&(outputFrames!=NULL)&&((gmm->nImgs/2) <= nFrames), OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);
	int offset=(gmm->outputHalfBuffer==0)?gmm->nImgs/2:0;
	for(int i=0; i<gmm->nImgs/2; i++)
	{
		//oclCheckErrorAndReturn((outputFrames[i]!=NULL), OCL_TRUE, funcname, pCleanUp);
		iReadFromOutputPinned(outputFrames[i], offset + i);

	}
#endif
#if __OVERLAP_COPY__==2
	oclCheckErrorAndReturn((gmm!=NULL)&&(outputFrames!=NULL)&&((gmm->nImgs) <= nFrames), OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);
	
	for(int i=0; i<gmm->nImgs/2; i++)
	{
		//oclCheckErrorAndReturn((outputFrames[i]!=NULL), OCL_TRUE, funcname, pCleanUp);
		iReadFromOutputPinned(outputFrames[i], i);

	}
#endif
	return status;
}

__IMPORTS int gmmPutInputFrames1(IplImage** inputFrames, int nFrames)
{
	const char* funcname="gmmPutInputFrames";
	int status=OCL_SUCCESS;
#if __OVERLAP_COPY__==2
	oclCheckErrorAndReturn((gmm!=NULL)&&(inputFrames!=NULL)&&(gmm->nImgs==nFrames), 
		OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);
	for(int i=0; i<gmm->nImgs; i++)
	{
		
		oclCheckErrorAndReturn((inputFrames[i]!=NULL)&&(inputFrames[i]->imageData!=NULL), OCL_TRUE, funcname, OCL_ERROR_NULL_PTR, pCleanUp);
		iWriteToInputPinned(inputFrames[i], i);
	}
#endif
#if __OVERLAP_COPY__==1
	oclCheckErrorAndReturn((gmm!=NULL)&&(inputFrames!=NULL)&&((gmm->nImgs/2)==nFrames), 
		OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);
	
	int offset=(gmm->inputHalfBuffer)?gmm->nImgs/2:0;
	for(int i=0; i<gmm->nImgs; i++)
	{
		
		oclCheckErrorAndReturn((inputFrames[i]!=NULL)&&(inputFrames[i]->imageData!=NULL), OCL_TRUE, funcname, OCL_ERROR_NULL_PTR, pCleanUp);
		iWriteToInputPinned(inputFrames[i], i);
	}
#endif
	return status;
}

#if __OVERLAP_COPY__==1
static int gmmPutInputFrames(CvCapture* capture)
{
	const char* funcname="igmmPutInputFrames";
	int status=OCL_SUCCESS;
	oclCheckErrorAndReturn((capture!=NULL)&&(gmm!=NULL), OCL_TRUE, funcname, OCL_ERROR_INVALID_ARGS, pCleanUp);

	IplImage* frame=NULL;
	
	int offset=(gmm->inputHalfBuffer==0)?gmm->nImgs/2:0;
	for(int i=0; i<gmm->nImgs/2; i++)
	{
		cvGrabFrame(capture);
		frame=cvRetrieveFrame(capture);
		oclCheckErrorAndReturn((frame!=NULL)&&(frame->imageData!=NULL), OCL_TRUE, funcname, OCL_ERROR_NULL_PTR, pCleanUp);
#ifdef __DEBUG__
		cvShowImage("input video", frame);
#endif
		iWriteToInputPinned(frame,offset + i);
		cvWaitKey(10);
	}
	return status;
}
#endif

__IMPORTS int gmmGetInfo(int gmmInfo)
{
	const char* funcName="gmmGetInfo";
	oclCheckErrorAndReturn(gmm!=NULL, OCL_TRUE, funcName, 0, pCleanUp);
	if(gmmInfo==GMM_INFO_NUM_FRAMES_PER_LOAD)
#if __OVERLAP_COPY__==1
		return gmm->nImgs/2;
#endif
#if __OVERLAP_COPY__==2
	return gmm->nImgs;
#endif
	if(gmmInfo==GMM_INFO_FRAME_WIDTH)
		return gmm->imgWidth;
	if(gmmInfo==GMM_INFO_FRAME_HEIGHT)
		return gmm->imgHeight;
	if(gmmInfo==GMM_INFO_FRAME_SIZE)
		return gmm->frameSize;
	if(gmmInfo==GMM_INFO_FRAME_DEPTH)
		return GMM_IMAGE_DEPTH;
	if(gmmInfo==GMM_INFO_INPUT_FRAME_NC)
		return GMM_HOST_INPUT_IMAGE_NUM_CHANELS;
	if(gmmInfo==GMM_INFO_OUTPUT_FRAME_NC)
		return GMM_HOST_OUTPUT_IMAGE_NUM_CHANELS;

	return GMM_INFO_ERROR;
}

#ifdef __GPU_TIMER__
__IMPORTS float gmmGetKernelExecTimeInMilliseconds()
{
	return gmm->executionTimeInMilliseconds;
}
__IMPORTS int gmmGetProcessedFrames()
{
	return gmm->frameCounter;
}
#endif