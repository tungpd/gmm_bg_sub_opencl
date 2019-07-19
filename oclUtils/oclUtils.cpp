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

#include "oclUtils.h"

//random intialize data
#ifdef __cplusplus
template<class T>
OCL_INLINE OCL_ERROR oclRandInitData(T** data, size_t dataSize)
{

}
#endif

//round up to the nearest multiple of the group_size
OCL_EXTERN_C int oclRoundUpGlobalSize(int group_size, int global_size)
{
	int r = global_size%group_size;
	if(r==0)
		return global_size;
	else
		return group_size + global_size - r; 
}
OCL_EXTERN_C cl_uint oclGetNumComputeUnits(cl_device_id deviceID, cl_int* error)
{
	//cl_int error=0;
	cl_uint max_compute_unit;
	*error=clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_unit, 0);
	
	if(*error!=CL_SUCCESS)
		return 0;
	return max_compute_unit;
}
OCL_EXTERN_C cl_ulong oclGetLocalMemSize(cl_device_id deviceID, cl_int* error)
{
	cl_ulong local_mem_size;
	*error=clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, 0);
	
	if(*error!=CL_SUCCESS)
		return 0;
	return local_mem_size;
}
OCL_EXTERN_C size_t oclGetMaxWorkGroupSize(cl_device_id deviceID, cl_int* error)
{
	size_t size;
	*error=clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, 0);
	if(*error!=CL_SUCCESS)
		return 0;
	return size;
}
OCL_EXTERN_C cl_uint oclGetMaxWorkItemDememsions(cl_device_id deviceID, cl_int* error)
{
	cl_uint dememsions;
	*error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dememsions, 0);
	if(*error!=CL_SUCCESS)
		return 0;
	return dememsions;
}
OCL_EXTERN_C cl_int oclGetMaxWorkItemSizes(cl_device_id deviceID, cl_uint numDememsions, size_t* size)
{
	cl_int error;
	error=clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, numDememsions*sizeof(size_t), size, 0);
	return error;
}
//Get Kernel source path from keyboard
OCL_EXTERN_C void oclGetKernelSourcePath(char* path, int num_chars)
{
	printf("Please enter the path of the OpenCL kernel source file: ");
#ifdef _WIN32 //windows version
	scanf_s("%s", path, num_chars);
#else
	scanf("%s", path);
#endif
	printf(path);
}

//auto fill float array @arr
OCL_EXTERN_C void oclFillFloatArray(float* arr, int size)
{
	float scale=1.0f/RAND_MAX;
	for(int j=0; j<size; j++)
	{
		arr[j]=scale*rand();
	}
}

//Load kernel sorce file
OCL_EXTERN_C int oclLoadKernelSourceEX(const char* path, char** kernelSourceString, long* kernelSourceSize)
{
	FILE *kernelSourceFile=NULL;
	char* __kernelSourceString=NULL;
	long __kernelSourceSize;
	//open kernel source file
#ifdef _WIN32	//windows version
	if(fopen_s(&kernelSourceFile, path, "rb")!=0)
	{
		__kernelSourceString=NULL;
		return -1;
	}
#else	//Linux version
	kernelSourceFile=fopen(path, "rb");
	if(kernelSourceFile==0)
	{
		__kernelSourceString=NULL;
		return;
	}
#endif
	//get size of kernel source
	fseek(kernelSourceFile, 0, SEEK_END);
	__kernelSourceSize=ftell(kernelSourceFile);
	fseek(kernelSourceFile, 0, SEEK_SET);

	//allocate buffer for __kernelSourceString and read the kernel source in
	__kernelSourceString=(char*)malloc(__kernelSourceSize + 1);
	if(fread(__kernelSourceString, __kernelSourceSize, 1, kernelSourceFile)!=1)
	{
		fclose(kernelSourceFile);
		free(__kernelSourceString);
		return -2;
	}
	__kernelSourceString[__kernelSourceSize]='\0';
	if(kernelSourceSize)
		*kernelSourceSize = __kernelSourceSize+1;
	fclose(kernelSourceFile);
	*kernelSourceString=__kernelSourceString;
	return 0;
//	if(!oclCheckError((path!=NULL), OCL_TRUE, "oclLoadKernelSourceEX"))
//		return OCL_ERROR_INVALID_ARGS;
//	if(!kernelSourceSize)
//	{
//		kernelSourceSize=(long*)malloc(sizeof(long));
//		*kernelSourceSize=0;
//	}
//
//#ifdef __cplusplus		//C++ version
//	std::fstream srcFile(path, std::fstream::in);
//	if(!srcFile.good())
//	{
//		std::cerr<<"ERROR: oclLoadKernelSourceEX(): Opening file failed."<<std::endl;
//		return OCL_ERROR_OPEN_FILE_FAILED;
//	}
//	char* __kernelSourceString=NULL;
//	long __kernelSourceSize;
//	srcFile.seekg(std::fstream::end);
//	__kernelSourceSize=srcFile.tellg();
//	srcFile.seekg(std::fstream::beg);
//
//	//allocate kernelSourceString
//	if(!((*kernelSourceString!=NULL) && (*kernelSourceSize==__kernelSourceSize)))
//	{
//		if(*kernelSourceSize!=__kernelSourceSize)
//		{
//			std::cerr<<"ERROR oclLoadKernelSourceEX(): kernelSourceSize not match!, reallocate kernelSourceString.....\n"<<std::endl;
//			//*kernelSourceSize=__kernelSourceSize;
//		}
//		if(*kernelSourceString != NULL) 
//			free(*kernelSourceString);
//		*kernelSourceString=(char*)malloc((__kernelSourceSize + 1)*sizeof(char));
//		//*kernelSourceSize=__kernelSourceSize + 1;
//	}
//	//read file
//	srcFile.read(*kernelSourceString, __kernelSourceSize);
//	//clean up
//	if(*kernelSourceString != NULL) 
//		free(kernelSourceString);
//	srcFile.close();
//	return OCL_SUCCESS;
//
//#else		//C version
//	FILE *kernelSourceFile=NULL;
//	char* __kernelSourceString=NULL;
//	long __kernelSourceSize;
//	//open kernel source file
//#ifdef _WIN32	//windows version
//	if(fopen_s(&kernelSourceFile, path, "rb")!=0)
//	{
//		__kernelSourceString=NULL;
//		return OCL_ERROR_OPEN_FILE_FAILED;
//	}
//#else	//Linux version
//	kernelSourceFile=fopen(path, "rb");
//	if(kernelSourceFile==0)
//	{
//		__kernelSourceString=NULL;
//		return OCL_ERROR_OPEN_FILE_FAILED;
//	}
//#endif
//	//get size of kernel source
//	fseek(kernelSourceFile, 0, SEEK_END);
//	__kernelSourceSize=ftell(kernelSourceFile);
//	fseek(kernelSourceFile, 0, SEEK_SET);
//
//	//allocate buffer for __kernelSourceString and read the kernel source in
//	__kernelSourceString=(char*)malloc((__kernelSourceSize + 1)*sizeof(char));
//	if(fread(__kernelSourceString, __kernelSourceSize, 1, kernelSourceFile)!=1)
//	{
//		fclose(kernelSourceFile);
//		if(__kernelSourceString!=NULL)
//			free(__kernelSourceString);
//		return OCL_ERROR_READ_FILE_FAILED;
//	}
//	__kernelSourceString[__kernelSourceSize]='\0';
//	if(kernelSourceSize)
//		*kernelSourceSize = __kernelSourceSize+1;
//	
//	*kernelSourceString=__kernelSourceString;
//	//clean up
//	fclose(kernelSourceFile);
//	if(__kernelSourceString)
//		free(__kernelSourceString);
//#endif
//	return OCL_SUCCESS;
}

//Get Device Execution Capabilites
OCL_EXTERN_C cl_device_exec_capabilities oclGetDevExecCap(cl_device_id deviceID, cl_int* error)
{
	cl_device_exec_capabilities devExecCap;
	*error=clGetDeviceInfo(deviceID, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(devExecCap), &devExecCap, 0);
	if(*error!=CL_SUCCESS)
		return (cl_device_exec_capabilities)0;
	return devExecCap;
}

//Display Platform information
OCL_EXTERN_C int oclDisplayPlatformInfo(cl_platform_id platformID, cl_platform_info platformInfo)
{
	cl_int error=CL_SUCCESS;
	int ok=1;
	
	//
	char* s;
	size_t n=0;

	error=clGetPlatformInfo(platformID, platformInfo, 0, 0, &n);
	if(error!=CL_SUCCESS)
	{
		printf("clGetPlatformInfo returned %d\n", error);
		ok=0;
	}
	else
	{
		++n;
		s=(char*)malloc(n);
		error=clGetPlatformInfo(platformID, platformInfo, n, s, 0);
		if(error!=CL_SUCCESS)
		{
			printf("clGetPlatformInfo returned %d\n", error);
			ok=0;
		}
		else
		{
			printf("\t\t\t----------------------------------------------\n");
			printf("\t\t\t%s\n\n",s);
		}
	}
	//clean up
	if(s) free(s);
	//
	return ok;
}

//get a number form keyboard
OCL_EXTERN_C int oclGetNumFromKeyboard()
{
	int i;
	fflush(stdin);
	scanf("%d", &i);
	return i;
}


//Get selected platform from keyboard.
OCL_EXTERN_C int oclGetSelectedPlatform()
{
	int selectedPlatform = 0;
	printf("please select a platform to compute (enter 0, 1, 2, ...): ");
	selectedPlatform = oclGetNumFromKeyboard();
	return selectedPlatform;
}

//display device information
OCL_EXTERN_C int oclDisplayDeviceInfo(cl_device_id deviceID, cl_device_info deviceInfo)
{
	cl_int error=CL_SUCCESS;
	int ok=1;

	size_t n=0;
	//char* s;
	
	error=clGetDeviceInfo(deviceID, deviceInfo, 0, 0, &n);
	if(error!=CL_SUCCESS)
	{
		printf("clGetDeviceInfo returned %d\n", error);
		ok=0;
	}
	else
	{
		if(deviceInfo == CL_DEVICE_MAX_WORK_ITEM_SIZES)
		{
			size_t *max_work_item_size=(size_t*)malloc(n);
			error=clGetDeviceInfo(deviceID, deviceInfo, n, max_work_item_size, 0);
			if(error!=CL_SUCCESS)
			{
				printf("clGetDeviceInfo returned %d\n", error);
				ok=0;
			}
			else
			{
				printf("\t----------------------------------------------\n");
				for(cl_uint i=0; i<n/sizeof(size_t); i++)
				{
					printf("\tmax_work_item_size[%d]=%u\n", i, max_work_item_size[i]);
				}
			}
			if(max_work_item_size) free(max_work_item_size);

		}
		if(deviceInfo==CL_DEVICE_NAME || deviceInfo==CL_DEVICE_VENDOR || deviceInfo==CL_DRIVER_VERSION 
			|| deviceInfo==CL_DEVICE_PROFILE || deviceInfo==CL_DEVICE_VERSION 
			|| deviceInfo == CL_DEVICE_OPENCL_C_VERSION || deviceInfo==CL_DEVICE_OPENCL_C_VERSION
			|| deviceInfo == CL_DEVICE_EXTENSIONS)
		{
			char* s=(char*)malloc(n);
			error=clGetDeviceInfo(deviceID, deviceInfo, n, s, 0);
			if(error!=CL_SUCCESS)
			{
				printf("clGetDeviceInfo returned %d\n", error);
				ok=0;
			}
			else
			{
				printf("\t----------------------------------------------\n");
				printf("\t%s\n\n",s);
			}
			if(s) free(s);
		}
		if(deviceInfo==CL_DEVICE_MAX_WORK_GROUP_SIZE || deviceInfo==CL_DEVICE_IMAGE2D_MAX_WIDTH
			|| deviceInfo==CL_DEVICE_IMAGE2D_MAX_HEIGHT|| deviceInfo==CL_DEVICE_IMAGE3D_MAX_WIDTH
			|| deviceInfo==CL_DEVICE_IMAGE3D_MAX_HEIGHT || deviceInfo==CL_DEVICE_IMAGE3D_MAX_DEPTH)
		{
			size_t size;
			error=clGetDeviceInfo(deviceID, deviceInfo, n, &size, 0);
			if(error!=CL_SUCCESS)
			{
				printf("clGetDeviceInfo returned %d\n", error);
				ok=0;
			}
			else
			{
				printf("\t----------------------------------------------\n");
				printf("\t%u\n\n",size);
			}
		}
		if(deviceInfo==CL_DEVICE_VENDOR_ID || deviceInfo==CL_DEVICE_MAX_COMPUTE_UNITS 
			|| deviceInfo==CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS || deviceInfo==CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
			|| deviceInfo==CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE || deviceInfo==CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
			|| deviceInfo==CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF || deviceInfo==CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
			|| deviceInfo==CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG || deviceInfo==CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
			|| deviceInfo==CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR || deviceInfo==CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
			|| deviceInfo== CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT || deviceInfo==CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
			|| deviceInfo==CL_DEVICE_NATIVE_VECTOR_WIDTH_INT || deviceInfo==CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
			|| deviceInfo==CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT || deviceInfo==CL_DEVICE_MAX_CLOCK_FREQUENCY
			|| deviceInfo==CL_DEVICE_MAX_READ_IMAGE_ARGS || deviceInfo==CL_DEVICE_MAX_WRITE_IMAGE_ARGS
			|| deviceInfo==CL_DEVICE_MAX_SAMPLERS || deviceInfo==CL_DEVICE_MEM_BASE_ADDR_ALIGN 
			|| deviceInfo==CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE || deviceInfo==CL_DEVICE_MAX_CONSTANT_ARGS)
		{
			cl_uint *info=(cl_uint*)malloc(n);
			error=clGetDeviceInfo(deviceID, deviceInfo, n, info, 0);
			if(error!=CL_SUCCESS)
			{
				printf("clGetDeviceInfo returned %d\n", error);
				ok=0;
			}
			else
			{
				printf("\t----------------------------------------------\n");
				printf("\t%u\n\n", info);
			}
			if(info) free(info);
		}

	}
	return ok;
}

//Get selected device from kerboard.
OCL_EXTERN_C int oclGetSelectedDevice()
{
	int selectedDevice=0;
	printf("please select a device to compute (enter 0, 1, 2, ...): ");
	selectedDevice=oclGetNumFromKeyboard();
	return selectedDevice;
}

//Display System information
OCL_EXTERN_C cl_int oclDisplaySystemInfo()
{
	cl_uint numPlatforms=0;
	cl_platform_id* platformIDs=0;
	cl_uint selectedPlatform=0;

	cl_uint numDevices=0;
	cl_device_id* deviceIDs=0;
	cl_uint selectedDevice=0;

	cl_int error=CL_SUCCESS;
	int ok=1;

	//Get Platforms
	if(ok==1)
	{
		printf("get platform id.....\n");
		error=clGetPlatformIDs(0, 0, &numPlatforms);
		if(error!=CL_SUCCESS)
		{
			printf("clGetPlatformIDs returned %d\n", error);
			ok=0;
		}
		else
		{
			platformIDs=(cl_platform_id*)malloc(numPlatforms*sizeof(*platformIDs));
			error=clGetPlatformIDs(numPlatforms, platformIDs, 0);
			if(error!=CL_SUCCESS)
			{
				printf("clGetPlatformIDs returned %d\n", error);
				ok=0;
			}
		}
	}
	//Display platform information
	printf("Platform information:\n");
	printf("##########################################################\n");
	printf("Number of platforms:\t%u\n", numPlatforms);
	for(cl_uint i=0; i<numPlatforms; i++)
	{
		printf("#%d\n", i);
		printf("\t************************************************\n");
		if(ok==1)
			ok=oclDisplayPlatformInfo(platformIDs[i], CL_PLATFORM_PROFILE);
		if(ok==1)
			ok=oclDisplayPlatformInfo(platformIDs[i], CL_PLATFORM_VERSION);
		if(ok==1)
			ok=oclDisplayPlatformInfo(platformIDs[i], CL_PLATFORM_NAME);
		if(ok==1)
			ok=oclDisplayPlatformInfo(platformIDs[i], CL_PLATFORM_VENDOR);
		/*if(ok==1)
			ok=oclDisplayPlatformInfo(platformIDs[i], CL_PLATFORM_EXTENSIONS);*/
	}

	if(platformIDs!=NULL) free(platformIDs);
	if(platformIDs!=NULL) free(deviceIDs);

	////get devive information
	//if(ok==1)
	//{
	//	printf("get the device information of platform #%u...............", selectedPlatform);
	//	error=clGetDeviceIDs(platformIDs[selectedPlatform], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
	//	if(error!=CL_SUCCESS)
	//	{
	//		printf("clGetDeviceIDs returned %d\n", error);
	//		ok=0;
	//	}
	//	else
	//	{
	//		deviceIDs=(cl_device_id*)malloc(numDevices*sizeof(*deviceIDs));
	//		error=clGetDeviceIDs(platformIDs[selectedDevice], CL_DEVICE_TYPE_ALL, numDevices, deviceIDs, 0);
	//		if(error!=CL_SUCCESS)
	//		{
	//			printf("clGetDeviceIDs returned %d\n", error);
	//			ok=0;
	//		}
	//	}
	//}
	////display device information
	//printf("#%d Device information", selectedDevice);
	//printf("#########################################################\n");
	//printf("Number of devices of the selelected device: %u\n", numDevices);
	//for(int i=0; i<numDevices; i++)
	//{
	//	printf("\t\t#%d\n", i);
	//	printf("\t\t***********************************************\n");
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_TYPE);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_VENDOR_ID);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_COMPUTE_UNITS);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_WORK_ITEM_SIZES);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_WORK_GROUP_SIZE);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT);
	//	if(ok)
	//		oclDisplayDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_CLOCK_FREQUENCY);

	//}
	return ok;
}
//Get platform id by name
OCL_EXTERN_C cl_platform_id oclGetPlatformByName(const char* platformName, cl_int *error)
{
	cl_platform_id platformID;
	cl_uint platformCount=0;
	cl_platform_id* platforms=0;
	char __platformName[128];
	//get number of platforms
	*error=clGetPlatformIDs(0, 0, &platformCount);
	if(*error != CL_SUCCESS)
		return (cl_platform_id)0;
	//alloc buffer for platforms
	platforms=(cl_platform_id*)malloc(platformCount*sizeof(*platforms));
	oclCheckError(platforms!=NULL, OCL_TRUE, "oclGetPlatformByName");
	//get all platforms
	*error=clGetPlatformIDs(platformCount, platforms, 0);
	if(*error != CL_SUCCESS)
	{
		//clean up 
		if(platforms) free(platforms);
		return (cl_platform_id)0;
	}
	//search for the platform have name platformName
	for(cl_uint i=0; i<platformCount; i++)
	{
		*error=clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128*sizeof(char), __platformName, 0);
		if(*error!=CL_SUCCESS)
			return (cl_platform_id)0;
		if(!strcmp(__platformName, platformName))
		{
			platformID=platforms[i];
			//clean up 
			if(platforms) free(platforms);

			return platformID;
		}
	}
	*error=OCL_ERROR_PLATFORM_NOT_FOUND;
	//printf("ERROR: Platform not found!\n\n");
	//clean up 
	if(platforms) free(platforms);
	return (cl_platform_id)0;
}

//Get build fail log
OCL_EXTERN_C OCL_ERROR oclDisplayBuildFailLog(cl_program program, cl_device_id deviceID)
{
	char* BuildInfoString=0;
	size_t BuildInfoStringSize;
	cl_int error;
	//init BuildInfoStringSize
	error=clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, 0, &BuildInfoStringSize);
	if(error!=CL_SUCCESS)
		return error;
	//alloc buffer for buildInfoString
	BuildInfoString=(char*)malloc((BuildInfoStringSize+1)*sizeof(char));
	
	if(!BuildInfoString)
	{
		printf("ERROR: alloc memory for BuildInfoString fail in function oclDisplayBuildFailLog.....\n\n");
		return OCL_ERROR_ALLOC_MEM_FAILED;
	}
	//get build info
	error=clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, BuildInfoStringSize, BuildInfoString, 0);
	if(error!=CL_SUCCESS)
	{
		printf("ERROR: %d in oclDisplayBuildFailLog()....\n\n", error);
		free(BuildInfoString);
		return error;
	}
	fflush(stdout);
	//display build info
	BuildInfoString[BuildInfoStringSize]='\0';
	printf("\n*****************************************************\n");
	printf("Build Log: \n");
	puts(BuildInfoString);
	printf("\n*****************************************************\n");
	//clean up
	free(BuildInfoString);

	return CL_SUCCESS;
}

//get build options from keyboard
OCL_EXTERN_C OCL_ERROR oclGetBuildOptions(char* buildOpts)
{
	const char* funcname="oclGetBuildOption";
	oclCheckError(buildOpts!=NULL, OCL_TRUE, funcname);
	fflush(stdin);
#ifdef WIN32

	printf_s("\n\nPlease enter build options:	");
	gets_s(buildOpts, 256); 
	printf_s("\n");
#else
	printf("\n\nPlease enter build options:	");
	gets(buildOpts); 
	printf("\n");
#endif
	return OCL_SUCCESS;
}
//save image to BMP format(windows only)
//@data data pointer
//@pixelSize size of pixel in bytes
//@width width of the image
//@height height of the image
//@fileName full name of BMP image file
#ifdef _WIN32
OCL_EXTERN_C OCL_ERROR oclSaveDataAsBMP(const char* data, size_t pixelSize, int width, int height, const char* fileName)
{
	//check args
	if(!oclCheckError(data!=NULL, OCL_TRUE, "oclSaveDataAsBMP")||!oclCheckError(pixelSize>0, OCL_TRUE, "oclSaveDataAsBMP")||
		!oclCheckError(width>0, OCL_TRUE, "oclSaveDataAsBMP")||!oclCheckError(height>0, OCL_TRUE, "oclSaveDataAsBMP")||
		!oclCheckError(fileName==NULL, OCL_TRUE, "oclSaveDataAsBMP"))
	{
		return OCL_ERROR_INVALID_ARGS;
	}
	FILE* file = 0;
	const int* ptr=(const int*)data;
	OCL_ERROR error=OCL_SUCCESS;

	printf("Save Image to: %s\n\n", fileName);
	//open file
	if(!oclCheckError(fopen_s(&file, fileName, "wb")==0, OCL_TRUE, "oclSaveDataAsBMP"))
		return OCL_ERROR_OPEN_FILE_FAILED;

	//create Bitmap file header
	BITMAPFILEHEADER	fileHeader;
	//create bitmap info header
	BITMAPINFOHEADER	infoHeader;

	//compute stride
	size_t stride = oclAlign((size_t)width*pixelSize, OCL_ALIGN_BMP_ROW);

	//initialize bitmap file header
	fileHeader.bfReserved1=0;
	fileHeader.bfReserved2=0;
	fileHeader.bfType = 0x4D42;
	//initialize bitmap info header
	infoHeader.biSize = sizeof(BITMAPINFOHEADER);
	infoHeader.biWidth = width;
	infoHeader.biHeight = height;
	infoHeader.biPlanes=1;
	infoHeader.biBitCount=32;
	infoHeader.biCompression=BI_RGB;
	infoHeader.biSizeImage=stride*height;
	infoHeader.biXPelsPerMeter=0;
	infoHeader.biYPelsPerMeter=0;
	infoHeader.biClrUsed=0;
	infoHeader.biClrImportant=0;

	//continue initialize bitmap file header
	fileHeader.bfSize=sizeof(fileHeader) + sizeof(infoHeader) + infoHeader.biSizeImage;
	fileHeader.bfOffBits=sizeof(fileHeader) + sizeof(infoHeader);

	//write fileheader to file
	if(oclCheckError(fwrite((void*)&fileHeader, sizeof(fileHeader), 1, file)!=sizeof(fileHeader), OCL_TRUE, "oclSaveDataAsBMP",)!=OCL_TRUE)
	{
		error = OCL_ERROR_WRITE_FILE_FAILED;
		goto exit;
	}
	//write infoheader to file
	if(oclCheckError(fwrite((void*)&infoHeader, sizeof(infoHeader), 1, file)!=sizeof(infoHeader), OCL_TRUE, "oclSaveDataAsBMP")!=OCL_TRUE)
	{
		error = OCL_ERROR_WRITE_FILE_FAILED;
		goto exit;
	}

	//create padding for each row
	unsigned char padding[4];
	for(int i=0; i<height; i++)
	{
		//seek the ptr the suitalbe row
		ptr=ptr + i*stride;

		for(int j; j<width; j++)
		{
			//write the pixel to the file
			if(!oclCheckError(fwrite((void*)ptr, sizeof(*ptr), 1, file)!=sizeof(*ptr), CL_TRUE, "oclSaveDataAsBMP"))
			{
				error=OCL_ERROR_WRITE_FILE_FAILED;
				goto exit;
			}
			ptr++;
		}
		//set padding
		memset(padding, 0x00, 4);
		if(oclCheckError(fwrite((void*)padding, 1, stride-(size_t)width*pixelSize, file)!=sizeof(*ptr), OCL_TRUE, "oclSaveDataAsBMP")!=OCL_TRUE)
		{
			error=OCL_ERROR_WRITE_FILE_FAILED;
			goto exit; 
		}
	}

	//clean up
	fclose(file);

	return OCL_SUCCESS;
exit:
	//clean up
	if(file!=NULL) fclose(file);
	return error;
}
#endif
//save image to PPM file
//@data handle to buffer of data
//@numChanels Number of chanels of the image that has one or three chanels, each chanel has size of one byte.
OCL_EXTERN_C OCL_ERROR oclSaveDataAsPPM(const char* data, int numChanels, int width, int height, const char* fileName)
{
	//check args
	if(!oclCheckError(data!=NULL&&&width>0&&height>0&&(numChanels==1||numChanels==3)&&
		fileName!=NULL, OCL_TRUE, "oclSaveDataAsPPM"))
		return OCL_ERROR_INVALID_ARGS;

	OCL_ERROR error=OCL_SUCCESS;
	//open file
#ifdef __cplusplus	//C++ version
	std::fstream stream(fileName, std::fstream::out|std::fstream::binary);
	//check error
	if(!oclCheckError(stream.good(), OCL_TRUE, "oclSaveDataAsPPM"))
		return OCL_ERROR_OPEN_FILE_FAILED;
	//write header
	if(numChanels==1)
		stream<<"P5\n";
	else
		stream<<"P6\n";
	stream<<width<<"\n"<<height<<"\n"<<0xff<<std::endl;
	//end write header

	//bigin write data to file
	for(int j=0; (j<width*height*numChanels)&&stream.good(); j++)
		stream<<data[j];
	if(!oclCheckError(stream.good(), OCL_TRUE, "oclSaveDataAsPPM"))
	{
		stream.close();
		return OCL_ERROR_WRITE_FILE_FAILED;
	}
	stream.flush();
	stream.close();

	return OCL_SUCCESS;


#else	//C version
	FILE* stream=NULL;
	stream=fopen(fileName, "wb");
	//check error
	if(!oclCheckError(stream!=NULL, OCL_TRUE, "oclSaveDataAsPPM"))
		return OCL_ERROR_OPEN_FILE_FAILED;
	//begin write header
	if(numChanels==1)
	{
		if(!oclCheckError(fwrite("P5\n", sizeof(char), 3, stream)==3, OCL_TRUE, "oclSaveDataAsPPM"))
		{
			fclose(stream);
			return OCL_ERROR_WRITE_FILE_FAILED;
		}
	}
	else
	{
		if(!oclCheckError(fwrite("P6\n", sizeof(char), 3, stream)==3, OCL_TRUE, "oclSaveDataAsPPM"))
		{
			fclose(stream);
			return OCL_ERROR_WRITE_FILE_FAILED;
		}
	}

	if(!oclCheckError(fprintf(stream,"%d\n%d\n%d\n", width, height, 0xff)>0, OCL_TRUE, "oclSaveDataAsPPM"))
	{
		fclose(stream);
		return OCL_ERROR_WRITE_FILE_FAILED;
	}
	//end write header

	//begin write data to file
	if(!oclCheckError(fwrite(data, sizeof(char), width*height*numChanels, stream)==width*height*numChanels,
		OCL_TRUE, "oclSaveDataAsPPM")
	{
		fclose(stream);
		return OCL_ERROR_WRITE_FILE_FAILED;
	}

	
	//end write data to file
	fclose(stream);
	return OCL_SUCCESS;
#endif

}
//Load image from PPM file
//
OCL_EXTERN_C OCL_ERROR oclLoadDataFromPPM(char** data, int* numChanels, int* width, int* height, const char* fileName)
{
	const char funcName[]="oclLoadDataFromPPM";
	//char imageType[OCL_STR_MAX_LEN];
	//check args
	if(!oclCheckError((data!=NULL)&&(numChanels!=NULL)&&(width!=NULL)&&(height!=NULL)&&(fileName!=NULL), OCL_TRUE, funcName))
	{
		return OCL_ERROR_INVALID_ARGS;
	}
	FILE* stream=0;
	//open file
	stream = fopen(fileName, "rb");
	if(!oclCheckError(stream!=NULL, OCL_TRUE, funcName))
	{
		return OCL_ERROR_OPEN_FILE_FAILED;
	}
	//check the header
	char header[OCL_STR_MAX_LEN];
	int counter=0;
	int w, h, max_pix_val, nChanels;
	//get header info
	while(counter<4)
	{
		if(!oclCheckError(fgets(header, OCL_STR_MAX_LEN, stream)!=NULL, OCL_TRUE, funcName))
		{
			fclose(stream);
			return OCL_ERROR_READ_FILE_FAILED;
		}
		if(header[0]=='#')
			continue;
		counter++;
		//get number of chanels
		if(counter==1)
		{
			if(strncmp(header, "P5", 2)==0)
				nChanels=1;
			else
			{
				if(strncmp(header, "P6", 2)==0)
					nChanels=2;
				else
				{
					printf("ERROR:#%d from %s (file format INVALID) at line %d in file %s\n\n",
						OCL_ERROR_FILE_FORMAT_INVALID, funcName, __LINE__, __FILE__);
					return OCL_ERROR_FILE_FORMAT_INVALID;
				}
			}
			continue;
		}
		//get width
		if(counter==2)
		{
			if(sscanf(header, "%u", &w)!=1||OCL_CHECK_OUT_OF_RANGE(w, OCL_IMAGE_WIDTH_MIN, OCL_IMAGE_WIDTH_MAX))
			{
				printf("ERROR: #%d(INVALID SIZE) from %s, at line %d in file %s\n\n", 
					OCL_ERROR_INVALID_SIZE, funcName, __LINE__, __FILE__);
				return OCL_ERROR_INVALID_SIZE;
			}
			continue;
		}
		//get height
		if(counter==3)
		{
			if(sscanf(header, "%u", &h)!=1||OCL_CHECK_OUT_OF_RANGE(h, OCL_IMAGE_HEIGHT_MIN, OCL_IMAGE_HEIGHT_MAX))
			{
				printf("ERROR: #%d(INVALID SIZE) from %s, at line %d in file %s\n\n", 
					OCL_ERROR_INVALID_SIZE, funcName, __LINE__, __FILE__);
				return OCL_ERROR_INVALID_SIZE;
			}
			continue;
		}
		//get max pixel value
		if(counter==4)
		{
			if(sscanf(header, "%u", &max_pix_val)!=1||OCL_CHECK_OUT_OF_RANGE(max_pix_val, OCL_IMAGE_PPM_MIN_PIX_VAL, OCL_IMAGE_PPM_MAX_PIX_VAL))
			{
				printf("ERROR: #%d(INVALID SIZE) from %s, at line %d in file %s\n\n", 
					OCL_ERROR_INVALID_VALUE, funcName, __LINE__, __FILE__);
				return OCL_ERROR_INVALID_VALUE;
			}
			continue;
		}
	}//end get header info

	//begin get data
	//
	//check if out put data buffer is allocated then check size if valid, use it
	//otherwise free the buffer and realloc.
	//if out put data buffer is not allicated then allocate buffer for out put data.
	//
	if(!((*data!=NULL)&&(*width==w)&&(*height==h)&&(*numChanels==nChanels)))
	{
		if(*data!=NULL) free(*data);
		*width=w;
		*height=h;
		*numChanels=nChanels;
		*data=(char*)malloc(w*h*nChanels*sizeof(char));
		if(!oclCheckError(data!=NULL, OCL_TRUE, funcName))
		{
			return OCL_ERROR_ALLOC_MEM_FAILED;
		}
	}
	//ok
	//read data from file
	int n;
	if(!oclCheckError(fread(*data, w*h*nChanels, 1, stream)==1, OCL_TRUE, funcName));
	{
		fclose(stream);
		return OCL_ERROR_READ_FILE_FAILED;
	}

	fclose(stream);
	return OCL_SUCCESS;
}
//*********************************************************************************************************************
//compare two float arrays (use L2-Norm) with an eps tolerence for equality
//return true if @ref and @data are identical, otherwise return false
//*********************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareL2NormEpsTolerance(const float* ref, const float* data, const unsigned int len, const float eps)
{
	oclCheckError(eps>=0, 1, "oclCompareL2NormEpsTolerance");
	float error=0;
	float ref2=0;
	for(unsigned int i = 0; i<len; ++i)
	{
		float diff=ref[i] - data[i];
		error+=diff*diff;
		ref2+=ref[i]*ref[i];
	}
	float normRef = sqrtf(ref2);

	if(normRef<=1e-7)
	{
#ifdef __DEBUG
		printf("ERROR ref L2-norm is 0\n");
#endif
		return OCL_FALSE;
	}
	//float normError=0;
	//normError = sqrtf(error);
	error=sqrtf(error)/normRef;
	return (OCL_BOOL) (error < eps);
}

#ifdef __cplusplus
//*********************************************************************************************************************
//compare two arrays of arbitrary type with an eps tolerence for equality and threshold for number of error pixels
//return true if @ref and @data are identical, otherwise return false
//@eps param use for comparison
//@thres param threshold % of number of pixel errors (ie. thres=1.5 it means 15% error pixel)
//*********************************************************************************************************************
template<class T>
OCL_BOOL oclCompareData(const T* ref, const T* data, const unsigned int len, const float eps, const float thres)
{
	oclCheckError(eps>=0, 1, "oclCompareData");
	//OCL_BOOL result = OCL_TRUE;
	unsigned int error_count=0;
	float diff;
	for(unsigned int i=0; i<len; i++)
	{
		diff = (float)(ref[i] - data[i]);
		if((diff <= eps && diff>=-eps)==OCL_FALSE)
			error_count++;
	}
	return error_count<=(unsigned int)(thres*(float)len)?OCL_TRUE: OCL_FALSE;
}
#else
#define OCL_IMPLEMENT_COMPARE_DATA(T)																						\
	OCL_BOOL oclCompareData##T(const T* ref, const T* data, const unsigned int len, const float eps, const float thres)		\
	{																														\
		oclCheckError(eps>=0, 1, "oclCompareData##T");																							\
		float diff;																											\
		unsigned int error_count=0;																							\
		for(int i=0; i<len; i++)																							\
		{																													\
			diff = (float)(ref[i] - data[i]);																				\
			if((diff <= eps && diff>=-eps)==OCL_FALSE)																		\
				error_count++;																								\
		}																													\
		return error_count<=(int)(thres*(float)len)?OCL_TRUE: OCL_FALSE;													\
	}
#endif

//*******************************************************************************************************************
//compare two float arrays
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareFloat(const float* ref, const float* data, const unsigned int lengh)
{
	return oclCompareData(ref, data, lengh, 0.0f, 0.0f);
}

//*******************************************************************************************************************
//compare two integer arrays
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareInt(const int* ref, const int* data, const unsigned int lengh)
{
	return oclCompareData(ref, data, lengh, 0, 0.0f);
}
//*******************************************************************************************************************
//compare two unsigned integer arrays, with epsilon and threshold
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareUint(const unsigned int* ref, const unsigned int* data, const unsigned int lengh, 
									const float eps, const float thresold)
{
	return oclCompareData(ref, data, lengh, eps, thresold);
}
//*******************************************************************************************************************
//compare two unsigned char arrays
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareUchar(const unsigned char* ref, const unsigned char* data, const unsigned int lengh)
{
	return oclCompareData(ref, data, lengh, 0.0f, 0.0f);
}

//*******************************************************************************************************************
//compare two unsigned char arrays (include Threshold for number of pixel we can have errors)
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareUcharThres(const unsigned char* ref, const unsigned char* data, const unsigned int lengh,
										const float eps, const float thres)
{
	return oclCompareData(ref, data, lengh, eps, thres);
}
//*******************************************************************************************************************
//compare two integer arrays with esp tolerance for equality
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareUcharEps(const unsigned char* ref, const unsigned char* data, const unsigned int lengh,
										const float eps)
{
	return oclCompareData(ref, data, lengh, eps, 0.0f);
}
//*******************************************************************************************************************
//compare two float arrays with esp tolerence for equality
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareFloatEps(const float* ref, const float* data, const unsigned int lengh, const float eps)
{
	return oclCompareData(ref, data, lengh, eps, 0.0f);
}
//*******************************************************************************************************************
//compare two float arrays with an eps tolerence for equality and a Threshold for number of pixel errors
//return true if @ref and @data are identical, otherwise return false
//*******************************************************************************************************************
OCL_EXTERN_C OCL_BOOL oclCompareFloatEpsThres(const float* ref, const float* data, const unsigned int lengh,
											const float eps, const float thres)
{
	return oclCompareData(ref, data, lengh, eps, thres);
}