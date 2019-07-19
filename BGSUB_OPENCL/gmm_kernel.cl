//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// This is an implementation of Zivkovic's Background Subtraction algorithm for GPU.
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


#define __OPENCL__
/* default parameters of gaussian background detection algorithm */
#define CV_BGFG_MOG2_STD_THRESHOLD            4.0f     /* lambda=2.5 is 99% */
#define CV_BGFG_MOG2_WINDOW_SIZE              500      /* Learning rate; alpha = 1/CV_GBG_WINDOW_SIZE */
#define CV_BGFG_MOG2_BACKGROUND_THRESHOLD     0.9f     /* threshold sum of weights for background test */
#define CV_BGFG_MOG2_STD_THRESHOLD_GENERATE   3.0f     /* lambda=2.5 is 99% */
#define CV_BGFG_MOG2_NGAUSSIANS               5        /* = K = number of Gaussians in mixture */
#define CV_BGFG_MOG2_VAR_INIT                 15.0f    /* initial variance for new components*/
#define CV_BGFG_MOG2_VAR_MIN                  4.0f
#define CV_BGFG_MOG2_VAR_MAX                  5*CV_BGFG_MOG2_VAR_INIT
#define CV_BGFG_MOG2_MINAREA                  15.0f    /* for postfiltering */

/* additional parameters */
#define CV_BGFG_MOG2_CT                       0.05f     /* complexity reduction prior constant 0 - no reduction of number of components*/
#define CV_BGFG_MOG2_SHADOW_VALUE             127       /* value to use in the segmentation mask for shadows, sot 0 not to do shadow detection*/
#define CV_BGFG_MOG2_SHADOW_TAU               0.5f      /* Tau - shadow threshold, see the paper for explanation*/

//define constant parameters
#define GMM_CONST_PARAMS_MAX_NUM_GAUSS			4
#define GMM_CONST_PARAMS_ALPHA					0.002f
#define GMM_CONST_PARAMS_SIGMA0					15.0f
#define GMM_CONST_PARAMS_Cthr					16.0f
#define GMM_CONST_PARAMS_CLOSE_THR				9.0f
#define GMM_CONST_PARAMS_ONE_MINUS_CF			0.9f
#define GMM_CONST_PARAMS_ALPHA_MUL_CT			0.0001f	//alpha*Ct

#define GMM_CONST_PARAMS_TAU					0.5f
#define GMM_CONST_PARAMS_SIGMA_MAX				75.0f
#define GMM_CONST_PARAMS_SIGMA_MIN				4.0f

#define GMM_CONST_PARAMS_SHADOW_VALUE			127
#define GMM_CONST_PARAMS_BG_VALUE				0
#define GMM_CONST_PARAMS_FG_VALUE				255


#ifdef __OPENCL__		//GPU version
//define data structure and operator of GMM params
#define GMMBgSubGoNextGaussMuSig(MuSigPtr, imgSz)					((MuSigPtr) + (imgSz))
#define GMMBgSubGetNextGaussMuSig(MuSigPtr, imgSz)					*GMMBgSubGoNextGaussMuSig(MuSigPtr, imgSz)
#define GMMBgSubGoPrevGaussMuSig(MuSigPtr, imgSz)					((MuSigPtr) - (imgSz))
#define GMMBgSubGetPrevGaussMuSig(MuSigPtr, imgSz)					*GMMBgSubGoPrevGaussMuSig(MuSigPtr, imgSz)

#define GMMBgSubGoNextGaussWeight(WeightPtr, imgSz)					((WeightPtr) + (imgSz))
#define GMMBgSubGetNextGaussWeight(WeightPtr, imgSz)				*GMMBgSubGoNextGaussWeight(WeightPtr, imgSz)
#define GMMBgSubGoPrevGaussWeight(WeightPtr, imgSz)					((WeightPtr) - (imgSz))
#define GMMBgSubGetPrevGaussWeight(WeightPtr, imgSz)				*GMMBgSubGoPrevGaussWeight(WeightPtr, imgSz)


#define GMMBgSubGoTonGauss(nGaussPtr, pix)							((nGaussPtr) + (pix))
#define GMMBgSubGetnGauss(nGaussPtr, pix)							*(GMMBgSubGoTonGauss(nGaussPtr, pix))

#ifndef __IMAGE_SIZE_POW2__
//Go to the MuSig params of the cGauss th Gaussian of the Mixture of Gaussians Model of the pix th
//Pt2FirstMuSig is a pointer that points to the Musig of the First Gaussian of the MoG model of 1st pixel.
#define GMMBgSubGoToGaussMuSig(cGauss, Pt2FirstMuSig, pix, imgSz)		((Pt2FirstMuSig) + (imgSz)*(cGauss) + (pix))
#define GMMBgSubGetGaussMuSig(cGauss, Pt2FirstMuSig, pix, imgSz)		*GMMBgSubGoToGaussMuSig(cGauss, Pt2FirstMuSig, pix, imgSz)
//Go to the weight of the cGauss th Gaussian of the Mixture of Gaussians Model of the pix th
//Pt2FirstWeight is a pointer that points to the weight of the First Gaussian of the MoG model of 1st pixel.
#define GMMBgSubGoToGaussWeight(cGauss, Pt2FirstWeight, pix, imgSz)		((Pt2FirstWeight) + (imgSz)*(cGauss) + (pix))
#define GMMBgSubGetGaussWeight(cGauss, Pt2FirstWeight, pix, imgSz)		*GMMBgSubGoToGaussWeight(cGauss, Pt2FirstWeight, pix, imgSz)

////Pt2FirstWeight is a pointer that points to the Musig of the First Gaussian of the MoG model of pix th pixel.
#define GMMBgSubGoToGaussMuSigPix(cGauss, Pt2FirstMuSig, imgSz)			((Pt2FirstMuSig) + (imgSz)*(cGauss))
#define GMMBgSubGetGaussMuSigPix(cGauss, Pt2FirstMuSig, imgSz)			*GMMBgSubGoToGaussMuSigPix(cGauss, Pt2FirstMuSig, imgSz)

//Pt2FirstWeight is a pointer that points to the weight of the First Gaussian of the MoG model of pix th pixel.
#define GMMBgSubGoToGaussWeightPix(cGauss, Pt2FirstWeight, imgSz)		((Pt2FirstWeight) + (imgSz)*(cGauss))
#define GMMBgSubGetGaussWeightPix(cGauss, Pt2FirstWeight, imgSz)		*GMMBgSubGoToGaussWeightPix(cGauss, Pt2FirstWeight, imgSz)
#else
//Go to the MuSig params of the cGauss th Gaussian of the Mixture of Gaussians Model of the pix th
//Pt2FirstMuSig is a pointer that points to the Musig of the First Gaussian of the MoG model of 1st pixel.
#define GMMBgSubGoToGaussMuSig(cGauss, Pt2FirstMuSig, pix, imgSzPow2)		((Pt2FirstMuSig) + (cGauss)<<(imgSzPow2) + (pix))
#define GMMBgSubGetGaussMuSig(cGauss, Pt2FirstMuSig, pix, imgSzPow2)		*GMMBgSubGoToGaussMuSig(cGauss, Pt2FirstMuSig, pix, imgSz)
//Go to the weight of the cGauss th Gaussian of the Mixture of Gaussians Model of the pix th
//Pt2FirstWeight is a pointer that points to the weight of the First Gaussian of the MoG model of 1st pixel.
#define GMMBgSubGoToGaussWeight(cGauss, Pt2FirstWeight, pix, imgSzPow2)		((Pt2FirstWeight) + (cGauss)<<(imgSzPow2) + (pix))
#define GMMBgSubGetGaussWeight(cGauss, Pt2FirstWeight, pix, imgSzPow2)		*GMMBgSubGoToGaussWeight(cGauss, Pt2FirstWeight, pix, imgSz)

////Pt2FirstWeight is a pointer that points to the Musig of the First Gaussian of the MoG model of pix th pixel.
#define GMMBgSubGoToGaussMuSigPix(cGauss, Pt2FirstMuSig, imgSzPow2)			((Pt2FirstMuSig) + (cGauss)<<(imgSzPow2))
#define GMMBgSubGetGaussMuSigPix(cGauss, Pt2FirstMuSig, imgSzPow2)			*GMMBgSubGoToGaussMuSigPix(cGauss, Pt2FirstMuSig, imgSz)

//Pt2FirstWeight is a pointer that points to the weight of the First Gaussian of the MoG model of pix th pixel.
#define GMMBgSubGoToGaussWeightPix(cGauss, Pt2FirstWeight, imgSzPow2)		((Pt2FirstWeight) + (cGauss)<<(imgSzPow2))
#define GMMBgSubGetGaussWeightPix(cGauss, Pt2FirstWeight, imgSzPow2)		*GMMBgSubGoToGaussWeightPix(cGauss, Pt2FirstWeight, imgSz)

#endif

#define GMMBgSubGoToFirstGaussMuSigPix(MuSigPtr)						(MuSigPtr)
#define GMMBgSubGoToFirstGaussWeightPix(weightPtr)						(weightPtr)
#define GMMBgSubGoToFirstGaussMuSig(MuSigPtr, offset, pix)				((MuSigPtr) + (pix))
#define GMMBgSubGoToFirstGaussWeight(weightPtr, offset, pix)			((weightPtr) + (pix))


#define __get_start_idx(x)			get_global_id(x)
#define __get_stride_size(x)		get_global_size(x)
#define __KERNEL					__kernel void
#define INLINE						inline
#else //CPU version

#endif

//This is an implementation of Zivkovic's Background Subtraction algorithm for GPU.
// For more details:
//		"Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction"
//			Z.Zivkovic, F. van der Heijden, Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.
//and
//		"Recursive unsupervised learning of finite mixture models"
//			Z.Zivkovic, F.van der Heijden, IEEE Trans. on Pattern Analysis and Machine Intelligence, vol.26, no.5, pages 651-656, 2004.
//
INLINE int __gmmBgSub(float red, float green, float blue, const int imgSz, 
						__global float4* MuSigPtr, __global float* weightPtr, __global int* nGaussPtr)
{
	int			isBackground = 0;
	int			isFitsPDF = 0;
	float		totalWeight = 0;

	float		oneMinusAlpha=1-GMM_CONST_PARAMS_ALPHA;

	__global float4* MuSigPtr0 = GMMBgSubGoToFirstGaussMuSigPix(MuSigPtr);
	__global float* weightPtr0 = GMMBgSubGoToFirstGaussWeightPix(weightPtr);
	int			nGauss = *nGaussPtr;
	//printf("nGauss=%d\n", nGauss);
	for(int cGauss=0; cGauss<nGauss; cGauss++)
	{
		float weight=*weightPtr0;//load weight from global memory

		weight=oneMinusAlpha*weight - GMM_CONST_PARAMS_ALPHA_MUL_CT;
		if(!isFitsPDF)
		{
			float4	MuSig=*MuSigPtr0;//load Musig from global memory
			float	dR = red - MuSig.x;
			float	dG = green - MuSig.y;
			float	dB = blue - MuSig.z;
			//float	Sigma = MuSig.w;
			float dist = (dR*dR + dB*dB + dG*dG);
			if((totalWeight < GMM_CONST_PARAMS_ONE_MINUS_CF)&&(dist< GMM_CONST_PARAMS_Cthr*MuSig.w))
					isBackground=1;
			if(dist < GMM_CONST_PARAMS_CLOSE_THR*MuSig.w)
			{
				isFitsPDF=1;
				weight += GMM_CONST_PARAMS_ALPHA;
				float k = GMM_CONST_PARAMS_ALPHA/weight;
				MuSig.x += k*dR;
				MuSig.y += k*dG;
				MuSig.z += k*dB;
				MuSig.w += k*(dist-MuSig.w);
				//limit the variance
				MuSig.w = MuSig.w < GMM_CONST_PARAMS_SIGMA_MIN ? GMM_CONST_PARAMS_SIGMA_MIN : 
					(MuSig.w > GMM_CONST_PARAMS_SIGMA_MAX?5*GMM_CONST_PARAMS_SIGMA_MAX:MuSig.w);
				//update parameters
				*MuSigPtr0=MuSig;
				*weightPtr0=weight;
				
				//sort
				__global float4* MuSigPtr1 = MuSigPtr0;
				__global float*	weightPtr1 = weightPtr0; 
				float tw=0.0f;
				for(int i = cGauss; (i>0)&&(weight>(tw=GMMBgSubGetPrevGaussWeight(weightPtr1, imgSz))); i--)
				{
					//swap parameters
					float4 tMusig;
					float  tweight;
					
					tMusig = *MuSigPtr1;
					tweight = *weightPtr1;
					
					*MuSigPtr1=GMMBgSubGetPrevGaussMuSig(MuSigPtr1, imgSz);
					*weightPtr1=tw;

					GMMBgSubGetPrevGaussMuSig(MuSigPtr1, imgSz) = tMusig;
					GMMBgSubGetPrevGaussWeight(weightPtr1, imgSz) = tweight;
					
					MuSigPtr1=GMMBgSubGoPrevGaussMuSig(MuSigPtr1, imgSz);
					weightPtr1=GMMBgSubGoPrevGaussWeight(weightPtr1, imgSz);
					
				}
			
			}
			else
				*weightPtr0=weight;
		}//
		else
			*weightPtr0=weight; //update new weight

		//decreate number of components if weight < 0
		if(weight < GMM_CONST_PARAMS_ALPHA_MUL_CT)
		{
			weight=0.0f;
			nGauss=cGauss;	//nGauss--
		}
		totalWeight += weight;
		MuSigPtr0 = GMMBgSubGoNextGaussMuSig(MuSigPtr0, imgSz);
		weightPtr0 = GMMBgSubGoNextGaussWeight(weightPtr0, imgSz);
	}//go through all modes
	
	//renormalize weights
	weightPtr0=GMMBgSubGoToFirstGaussWeightPix(weightPtr);
	for(int i=0; i<nGauss; i++)
	{
		*weightPtr0 /= totalWeight;
		weightPtr0=GMMBgSubGoNextGaussWeight(weightPtr0, imgSz);
	}
	
	
	//make new mode if needed and exit
	if(!isFitsPDF)
	{
		
		if(nGauss<GMM_CONST_PARAMS_MAX_NUM_GAUSS)
			nGauss++;
		//GMMBgSubGetGaussWeightPix(nGauss - 1, weightPtr, imgSz)=GMM_CONST_PARAMS_ALPHA;
		
		float4 tMuSig;
		
		tMuSig.x=red;
		tMuSig.y=green;
		tMuSig.z=blue;
		tMuSig.w=GMM_CONST_PARAMS_SIGMA0;

		//GMMBgSubGetGaussMuSigPix(nGauss-1, MuSigPtr, imgSz)=tMuSig;
		
		//sort
		__global float4* MuSigPtr1 = GMMBgSubGoToGaussMuSigPix(nGauss-1, MuSigPtr, imgSz);
		__global float*	weightPtr1 = GMMBgSubGoToGaussWeightPix(nGauss-1, weightPtr, imgSz);
		*MuSigPtr1=tMuSig;
		*weightPtr1=(nGauss==1) ? 1:GMM_CONST_PARAMS_ALPHA;

		float tw=0.0f;
		for(int i = nGauss-1; (i>0)&&(GMM_CONST_PARAMS_ALPHA>(tw=GMMBgSubGetPrevGaussWeight(weightPtr1, imgSz))); i--)
		{
			//swap parameters
			float4 tMusig;
			float  tweight;
					
			tMusig = *MuSigPtr1;
			tweight = *weightPtr1;
					
			*MuSigPtr1=GMMBgSubGetPrevGaussMuSig(MuSigPtr1, imgSz);
			*weightPtr1=tw;

			GMMBgSubGetPrevGaussMuSig(MuSigPtr1, imgSz) = tMusig;
			GMMBgSubGetPrevGaussWeight(weightPtr1, imgSz) = tweight;
					
			MuSigPtr1=GMMBgSubGoPrevGaussMuSig(MuSigPtr1, imgSz);
			weightPtr1=GMMBgSubGoPrevGaussWeight(weightPtr1, imgSz);
		}

		//renormalize weights if nGauss>1
		if(nGauss>1)
		{
			weightPtr0=GMMBgSubGoToFirstGaussWeightPix(weightPtr);
			for(int i=0; i<nGauss-1; i++)
			{
				(*weightPtr0)*=oneMinusAlpha;//
				weightPtr0=GMMBgSubGoNextGaussWeight(weightPtr0, imgSz);
			}
		}
		
	}
	
	*nGaussPtr=nGauss;	//update number of Gaussians
	
	return isBackground;
}
//detect shadow
//For more information see: 
//     Andrea Prati, Ivana Mikic, Mohan M. Trivedi, Rita Cucchiara, "Detecting Moving Shadows: Algorithms and Evaluation" IEEE PAMI,2003.
//
INLINE int gmmRemoveShadow(float red, float green,  float blue, const int imgSz, __global const float4* MuSigPtr, __global const float* weightPtr, int nGauss)
{
	float totalWeight=0;
	float numerator, denominator;
	//float4* MuSigPtr=0;
	//float*	weightPtr=0;
	__global const float4* MuSigPtr0=GMMBgSubGoToFirstGaussMuSigPix(MuSigPtr);
	__global const float* weightPtr0=GMMBgSubGoToFirstGaussWeightPix(weightPtr);
	for(int i=0; (totalWeight<GMM_CONST_PARAMS_ONE_MINUS_CF)&&(i<nGauss); i++)
	{
		const float4	MuSig=*MuSigPtr0;
		const float		weight=*weightPtr0;
		totalWeight+=weight;

		numerator=red*MuSig.x + green*MuSig.y + blue*MuSig.z;
		denominator=MuSig.x*MuSig.x + MuSig.y*MuSig.y + MuSig.z*MuSig.z;
		if(denominator==0)
			break;

		float a=numerator/denominator;
		if((a<=1)&&(a>=GMM_CONST_PARAMS_TAU))
		{
			float dR=a*MuSig.x - red;
			float dG=a*MuSig.y - green;
			float dB=a*MuSig.z - blue;

			float dist=dR*dR + dG*dG + dB*dB;
			if(dist<GMM_CONST_PARAMS_Cthr*MuSig.w*a*a)
				return 2;

		}
		//go next gaussian
		MuSigPtr0=GMMBgSubGoNextGaussMuSig(MuSigPtr0, imgSz);
		weightPtr0=GMMBgSubGoNextGaussWeight(weightPtr0, imgSz);
	}
	return 0;
}

#ifdef __PROC_MULTIFRAMES__
__KERNEL gmmBgSub(__global const char* inputImages, __global char* outputImages, int nImages,
					 int inputOffset, int outputOffset, int ImageSize, int inputHalfBuffer, int outputHalfBuffer,
						__global float4* MuSigPtr, __global float* weightPtr, __global int* nGaussianPtr, int detectShadow)
{
	
	__global const char4* inputImage = (__global const char4*)(inputImages + inputHalfBuffer);
	__global char* outputImage = (__global char*)(outputImages + outputHalfBuffer);
	
	for(int imgCount=0; imgCount<nImages; imgCount++)
	{
		for(int pix=__get_start_idx(0); pix<ImageSize; pix+=__get_stride_size(0))
		{
			char4 curPix;
			curPix=inputImage[pix];
			
			__global int* nGaussianPtr0=GMMBgSubGoTonGauss(nGaussianPtr, pix);
			__global float4* MuSigPtr0=GMMBgSubGoToFirstGaussMuSig(MuSigPtr, 0, pix);
			__global float* weightPtr0=GMMBgSubGoToFirstGaussWeight(weightPtr, 0, pix);
		
			int result=__gmmBgSub(curPix.x, curPix.y, curPix.z, ImageSize, MuSigPtr0, weightPtr0, nGaussianPtr0);
			
			if((!result)&&detectShadow)
				result=gmmRemoveShadow(curPix.x, curPix.y, curPix.z, ImageSize, MuSigPtr0, weightPtr0, *nGaussianPtr0);
			switch(result)
			{
			case 0://foreground
				outputImage[pix]=255;
				break;
			case 1://backround
				outputImage[pix]=0;
				break;
			case 2: //shadow
				outputImage[pix]=128;
				break;

			
		}

		//go to next input image(RGBA image), imageSize is the size of input image in pixels
		inputImage = (__global const char4*)(((__global const char*)inputImage) + inputOffset);
		//go to next output image
		outputImage = outputImage + outputOffset;
	}
}
#else
__KERNEL gmmBgSub(__global const char* inputImage, __global char* outputImage, /*int nImages,*/
					 /*int inputOffset, int outputOffset,*/ int ImageSize, int inputHalfBuffer, int outputHalfBuffer,
						__global float4* MuSigPtr, __global float* weightPtr, __global int* nGaussianPtr, int detectShadow)
{
	
	__global const char4* __inputImage = (__global const char4*)(inputImage + inputHalfBuffer);
	__global char* __outputImage = (__global char*)(outputImage + outputHalfBuffer);
	
	/*for(int imgCount=0; imgCount<nImages; imgCount++)
	{*/
		for(int pix=__get_start_idx(0); pix<ImageSize; pix+=__get_stride_size(0))
		{
			char4 curPix;
			curPix=__inputImage[pix];
			
			__global int* nGaussianPtr0=GMMBgSubGoTonGauss(nGaussianPtr, pix);
			__global float4* MuSigPtr0=GMMBgSubGoToFirstGaussMuSig(MuSigPtr, 0, pix);
			__global float* weightPtr0=GMMBgSubGoToFirstGaussWeight(weightPtr, 0, pix);
		
			int result=__gmmBgSub(curPix.x, curPix.y, curPix.z, ImageSize, MuSigPtr0, weightPtr0, nGaussianPtr0);
			
			if((!result)&&detectShadow)
				result=gmmRemoveShadow(curPix.x, curPix.y, curPix.z, ImageSize, MuSigPtr0, weightPtr0, *nGaussianPtr0);
			switch(result)
			{
			case 0://foreground
				__outputImage[pix]=255;
				break;
			case 1://backround
				__outputImage[pix]=0;
				break;
			case 2: //shadow
				__outputImage[pix]=128;
				break;
			}
			
		}

		//go to next input image(RGBA image), imageSize is the size of input image in pixels
		//inputImage = (__global const char4*)(((__global const char*)inputImage) + inputOffset);
		//go to next output image
		//outputImage = outputImage + outputOffset;
	/*}*/
}
#endif
