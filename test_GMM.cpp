#include <oclUtils.h>
#include <GMMBgSub.h>
#include <CL\cl.h>
#include <cv.h>
#include <highgui.h>



int main()
{
	const char* funcName="main";
	CvCapture* capture;
	//capture=cvCaptureFromFile("D:\\output2.avi");
	capture=cvCaptureFromCAM(0);
	assert(capture!=NULL);

	//cvNamedWindow("Input");
	cvNamedWindow("Output");

	int status;
	//init gmm
	status = gmmInitGMMBgSub((int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT));
	int frameRate=(int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	printf("\n\nNumber of frames per second of input video: %d\n", frameRate);
	if(!oclCheckError(status, OCL_SUCCESS, funcName))
		cvReleaseCapture(&capture);

	int nImages=gmmGetNumFrames();
	int frameWidth=gmmGetFrameWidth();
	int frameHeight=gmmGetFrameHeight();
	int frameDepth=gmmGetFrameDepth();
	int outputFrameNChannels=gmmGetOutpuFrameNumChannels();
	//Create output frames
	IplImage** outputFrames=(IplImage**)malloc(nImages*sizeof(IplImage*));
	assert(outputFrames!=NULL);
	for(int i=0; i<nImages; i++)
	{
		outputFrames[i]=cvCreateImage(cvSize(frameWidth, frameHeight), frameDepth, outputFrameNChannels);
	}

	//int outputImageSize=outputFrames[0]->height*outputFrames[0]->widthStep;
	//IplImage* outputImageRef=gmmCreateTestOutputImage(cvSize(frameWidth, frameHeight), 2*25);
	//IplImage* frame=NULL;
	//frame=cvQueryFrame(capture);
	//cvShowImage("Input", frame);
	//cvWaitKey(5);

	////create input frames for testing
	//IplImage** inputImages=(IplImage**)malloc(nImages*sizeof(IplImage*));
	//for(int i=0; i<nImages; i++)
	//{
	//	inputImages[i]=gmmCreateTestInputImage(cvSize(frameWidth, frameHeight), 25);
	//}
	//char c;
	int wtime=1000/frameRate;
	while(1)
	{
		//gmmPutInputFrames(inputImages, nImages);

		if(gmmUpdateGMMBgSub(capture)!=OCL_SUCCESS)
			exit(-1);
		gmmGetOutputFrames(outputFrames, nImages);
		for(int i=0; i<nImages; i++)
		{	
			cvShowImage("Output", outputFrames[i]);
			cvWaitKey(10);
		}
		printf("Number of frames per second: %f\n", 1000.0f*(float)gmmGetProcessedFrames()/gmmGetKernelExecTimeInMilliseconds());
		
	}
	cvWaitKey(100);
	cvReleaseCapture(&capture);
	
	return 0;
}