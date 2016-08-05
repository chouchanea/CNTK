//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalClient.cpp : Sample application using the evaluation interface from C++
//

#include "stdafx.h"
#include "eval.h"
#include "mex.h"
#include "matrix.h"
#include "stdint.h";

using namespace Microsoft::MSR::CNTK;
#include <iostream>
using namespace std;

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

int _tmain(int argc, wchar_t* argv[])
{

	return 0;
}
/* The gateway function */
std::vector<float> imageToVector(double *, mwSize, mwSize, mwSize);

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{

	

	std::string func = "GetEvalF";
	auto hModule = LoadLibrary(_T("evaldll.dll"));
	auto procAddress = GetProcAddress(hModule, func.c_str());
	auto getEvalProc = (GetEvalProc<float>)procAddress;
	IEvaluateModel<float> *model;
	getEvalProc(&model);

	//model->CreateNetwork("modelPath=\"C:\\CNTK_binary\\cntk\\Examples\\Image\\MNIST\\Output\\Models\\02_Convolution\"");
	mexPrintf("Creating nework...\n");

	//original
	//model->CreateNetwork("modelPath=\"C:\\CNTK_binary\\cntk\\Examples\\Image\\Miscellaneous\\CIFAR-10\\Output\\Models\\01_Convolution\"");

	//Amir
	model->CreateNetwork("modelPath=\"C:\\Users\\cho\\Downloads\\CNTK-master\\Examples\\Image\\MNIST\\Output\\Models\\01_OneHidden\"");

	std::map<std::wstring, size_t> inDims;
	std::map<std::wstring, size_t> outDims;
	model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
	model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);

	//Generate Dummy input values 
	auto inputLayerName = inDims.begin()->first;

	//Amir
	cout << inDims.begin()->second << endl;

	//Creating an input from 2D image so that we could transofrm it to 1D vector
	double *inImage;
	mwSize nrows, ncols, ndims;
	const mwSize *dims;
	mwSize nchannels;

	//Amir
	//uint8_t *testing;
	//testing = (uint8_t *)mxGetData(prhs[0]);
	//inImage = (double *)testing;

    inImage = mxGetPr(prhs[0]);
	ndims = mxGetNumberOfDimensions(prhs[0]);
	dims = mxGetDimensions(prhs[0]);

	nrows = dims[0];
	ncols = dims[1];


	if (ndims == 3){
		nchannels = dims[2];
	}
	else {
		nchannels = 1;
	}
	mexPrintf("Number of dimensions: %d\n", ndims);
	for (int i = 0; i < ndims; i++) {
		mexPrintf("Dimension %d : %d\n", i, dims[i]);
	}
	mwSize inputDimProduct = (nrows * ncols * nchannels);
	mexPrintf("Number of input dimensions: %d\n", inDims.begin()->second);
	mexPrintf("Product of input dimensions: %d\n", inputDimProduct);

	//important: UNCOMMENT
	if (inputDimProduct != inDims.begin()->second) {
		mexErrMsgIdAndTxt("MATLAB:mxcalcsinglesubscript:inputMismatch",
			"Product of image dimensions does not comply with the network input dimensionality");
	}

	if (ndims > 3)  {
		mexErrMsgIdAndTxt("MATLAB:mxcalcsinglesubscript:inputMismatch",
			"Incorrect number of dimensions - it might be either 1 or 3 - graysscale or RGB image");
	}

	if ((ndims == 3) && (dims[2] == 2)){
		mexErrMsgIdAndTxt("MATLAB:mxcalcsinglesubscript:inputMismatch",
			"Incorrect number of dimensions - it might be either 1 or 3 - graysscale or RGB image");
	}
	

	//Checking data hand-over from Matlab
	cout << mxIsDouble(prhs[0]) << endl;

	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++){
			mexPrintf(" Input (%d, %d): [ %f  ]\n", i, j, inImage[i + j*nrows]);
			//cout << "Grayscale " << j << ", " << i << " :" << inImage[i + j*nrows] << endl;
		}
	}


	std::vector<float> inputs = imageToVector(inImage, nrows, ncols, nchannels);
	std::vector<float> outputs;

	// Setup the maps for inputs and output
	Layer inputLayer;
	inputLayer.insert(MapEntry(inputLayerName, &inputs));
	Layer outputLayer;
	auto outputLayerName = outDims.begin()->first;
	outputLayer.insert(MapEntry(outputLayerName, &outputs));



	//Amir //Just testing
	//model->StartEvaluateMinibatchLoop();


	model->Evaluate(inputLayer, outputLayer);

	// Output the results
	double *outputResponses;
	plhs[0] = mxCreateDoubleMatrix(1, outDims.begin()->second, mxREAL);
	outputResponses = mxGetPr(plhs[0]);
	int i = 0;
	for each (auto& value in outputs)
	{
		cout << "Output: " << value << endl;
		outputResponses[i++] = value;
	}

	//Amir
	cout << "Done." << endl;

	return;
}

std::vector<float> imageToVector(double *inImage, mwSize nrows, mwSize ncols, mwSize nchannels){
	/*Converts given image to a linear vector*/
	std::vector<float> inputs;
	if (nchannels == 1) {
		mexPrintf("Converting graysscale image to vector...\n");
		/*Working with graysacale image*/
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < ncols; j++){
				double input = inImage[i + j*nrows];
				//cout << "Grayscale " << j << ", " << i << " :" << inImage[i + j*nrows] << endl;
				inputs.push_back(input);
				//mexPrintf("Input: %f ", input);
				//cout << endl;
			}
		}
	}
	else
	{
		/*Working with RGB image*/
		mexPrintf("Converting RGB image to vector...\n");
		for (int nc = 2; nc >= 0; nc--) {
			for (int i = 0; i < nrows; i++)	{
				for (int j = 0; j < ncols; j++) {
					double input = inImage[i + j*nrows + nc*(nrows * ncols)];
					inputs.push_back(input);
				}
			}
		}
	}
	return inputs;
}