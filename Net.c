
#pragma once

#include "Net.h"
#include <stdlib.h>

net_dataType* net_InitiateBlank(int length) {

	return calloc(length, sizeof(net_dataType));

};


net_dataType* net_InitiateRandom(int length, int negativesAllowed) {

	net_dataType* temp = calloc(length, sizeof(net_dataType));

	for (int i = 0; i < length; i++) {

		temp[i] = (net_dataType)((rand() % 200) / 100.0 - (negativesAllowed * 1));
	};

	return temp;

};


void net_DisplayModel(net_dataType* net_weights, net_dataType* net_bias,
	int nIn, int nWidth, int nDepth, int nOut) {

	printf("\nModel\n");

	printf("      [Inputs..]");

	printf("\n");

	int iWeight = 0;
	int iBias = 0;

	printf("w0    ");
	for (int i = 0; i < nIn * nWidth; i++) {
		printf("%-5.2f ", net_weights[iWeight]);
		iWeight++;
	};

	printf("\n");

	printf("b0    ");
	for (int j = 0; j < nWidth; j++) {
		printf("%-5.2f ", net_bias[iBias]);
		iBias++;
	};

	printf("\n");

	for (int i = 0; i < nDepth - 1; i++) {

		printf("w%-5d", i + 1);
		for (int j = 0; j < nWidth * nWidth; j++) {
			printf("%-5.2f ", net_weights[iWeight]);
			iWeight++;
		};

		printf("\n");

		printf("b%-5d", i + 1);
		for (int j = 0; j < nWidth; j++) {
			printf("%-5.2f ", net_bias[iBias]);
			iBias++;
		};

		printf("\n");

	};

	printf("w%-5d", nDepth);
	for (int i = 0; i < nWidth * nOut; i++) {
		printf("%-5.2f ", net_weights[iWeight]);
		iWeight++;
	};

	printf("\n");

	printf("b%-5d", nDepth);
	for (int i = 0; i < nOut; i++) {
		printf("%-5.2f ", net_bias[iBias]);
		iBias++;
	};

	printf("\n\n");

	return;

};


void net_DisplayActivations(net_dataType* net_inputs, net_dataType* net_activations, net_dataType* net_outputs,
	int nIn, int nWidth, int nDepth, int nOut) {

	printf("\nActivations\n");

	printf("i      ");
	for (int i = 0; i < nIn; i++) {
		printf("%-6.2f ", (float)net_inputs[i]);
	};

	printf("\n");

	for (int i = 0; i < nDepth; i++) {

		printf("l%-6d", i);
		for (int j = 0; j < nWidth; j++) {
			printf("%-6.2f ", (float)net_activations[i * nWidth + j]);
		};

		printf("\n");

	};

	printf("o      ", nDepth);
	for (int i = 0; i < nOut; i++) {
		printf("%-6.2f ", (float)net_outputs[i]);
	};

	printf("\n\n");

	return;

};


void net_DisplayGradient(net_dataType* net_weightGradient, net_dataType* net_biasGradient,
	int nIn, int nWidth, int nDepth, int nOut) {

	printf("\nGradient\n");

	printf("\n");

	int iWeight = 0;
	int iBias = 0;
	for (int i = 0; i < nDepth + 1; i++) {

		int wSize = nWidth * nWidth;
		if (i == 0) {
			wSize = nIn * nWidth;
		};
		if (i == nDepth) {
			wSize = nWidth * nOut;
		};

		int bSize = nWidth;
		if (i == nDepth) {
			bSize = nOut;
		};

		printf("w%-5d", i);
		for (int j = 0; j < wSize; j++) {
			printf("%c", 176 * (net_weightGradient[iWeight] < 0) + 178 * (net_weightGradient[iWeight] > 0) + ' ' * (net_weightGradient[iWeight] == 0));
			iWeight++;
		};

		printf("\n");

		printf("b%-5d", i);
		for (int j = 0; j < bSize; j++) {
			printf("%c", 177 - (net_biasGradient[iBias] < 0) + (net_biasGradient[iBias] > 0));
			iBias++;
		};

		printf("\n");

	};

	printf("\n");

	return;

};


double net_ActivationFunction(double activation) {

	if (activation < 0) {
		return (activation * .01);
	};

	return activation;

};

double net_dActivationFunction(double activation) {

	if (activation < 0) {
		return -.01;
	};

	return 1.0;

};

int net_Infer(
	net_dataType* net_inputs, net_dataType* net_weights, net_dataType* net_bias,
	net_dataType* net_activations, net_dataType* net_zActivations, net_dataType* net_outputs,
	int nIn, int nWidth, int nDepth, int nOut) {

	int iWeight = 0;
	int iActivation = 0;
	int iBias = 0;
	
	for (int i = 0; i < nWidth; i++) {

		double sum = 0;

		for (int j = 0; j < nIn; j++) {

			sum += net_weights[iWeight] * net_inputs[j];

			iWeight++;

		};

		double activation = sum + net_bias[iBias];

		net_zActivations[iActivation + i] = activation;

		// inputLayer activation fx
		net_ActivationFunction(activation);

		net_activations[iActivation + i] = activation;

		iBias++;

	};

	iActivation += nWidth;

	for (int i = 0; i < nDepth - 1; i++) {

		for (int j = 0; j < nWidth; j++) {

			double sum = 0;

			for (int k = 0; k < nWidth; k++) {

				sum += net_weights[iWeight] * net_activations[iActivation - nWidth + k];

				iWeight++;

			};

			double activation = sum + net_bias[iBias];

			net_zActivations[iActivation + j] = activation;

			// midLayer activation fx
			net_ActivationFunction(activation);

			net_activations[iActivation + j] = activation;

			iBias++;

		};

		iActivation += nWidth;

	};

	for (int i = 0; i < nOut; i++) {

		double sum = 0;

		for (int j = 0; j < nWidth; j++) {

			sum += net_weights[iWeight] * net_activations[iActivation - nWidth + j];

			iWeight++;

		};

		double activation = sum + net_bias[iBias];

		net_zActivations[iActivation + i] = activation;

		// outputLayer activation fx
		net_ActivationFunction(activation);

		net_activations[iActivation + i] = activation;
		net_outputs[i] = net_activations[iActivation + i];

		iBias++;

	};

	return 1;
};


int net_Backprop(
	net_dataType* net_inputs, net_dataType* net_weights, net_dataType* net_bias,
	net_dataType* net_activations, net_dataType* net_zActivations,
	int nIn, int nWidth, int nDepth, int nOut,
	net_dataType* net_desiredOutputs,
	net_dataType* net_weightGradient,
	net_dataType* net_biasGradient,
	double learnRate) {

	int nWeights = net_nWeights(nIn, nWidth, nDepth, nOut);
	int nBias = net_nBias(nWidth, nDepth, nOut);
	int nActivations = nBias;

	int jBias = nBias - 1;
	int jActivation = nActivations - 1;
	int jkWeight = nWeights - 1;
	int kActivation = nActivations - nOut - 1;

	net_dataType* betterActivations = net_activations;

	for (int i = 0; i < nOut; i++) {
		
		double a = net_desiredOutputs[nOut - i - 1] - net_ActivationFunction(net_zActivations[jActivation]);
		double b = -net_dActivationFunction(net_zActivations[jActivation]);

		double prod = a * b * learnRate;

		net_biasGradient[jBias] -= prod;

		for (int j = 0; j < nWidth; j++) {

			double x = net_activations[jActivation];

			if (x > 5) {
				x = 5;
			};
			if (x < -5) {
				x = -5;
			};

			double w = net_weights[jkWeight];

			net_weightGradient[jkWeight] -= prod * x;

			jkWeight--;

			betterActivations[kActivation - j] -= prod * w;

		};

		jActivation--;
		jBias--;

	};

	kActivation -= nWidth;

	for (int i = 0; i < nDepth - 1; i++) {
		
	for (int j = 0; j < nWidth; j++) {
		
		double a = betterActivations[jActivation] - net_ActivationFunction(net_zActivations[jActivation]);
		double b = -net_dActivationFunction(net_zActivations[jActivation]);

		double prod = -a * b * learnRate;

		net_biasGradient[jBias] -= prod;

		double x = net_activations[jActivation];

		if (x > 5) {
			x = 5;
		};
		if (x < -5) {
			x = -5;
		};

		double w = 0;

		for (int k = 0; k < nWidth; k++) {

			double x = net_activations[jActivation];

			double w = net_weights[jkWeight];

			net_weightGradient[jkWeight] -= prod * x;

			jkWeight--;

			betterActivations[kActivation - k] -= prod * w;

		};

		jBias--;
		jActivation--;

	};

	kActivation -= nWidth;

	};

	for (int i = 0; i < nWidth; i++) {

		double a = betterActivations[jActivation] - net_ActivationFunction(net_zActivations[jActivation]);
		double b = -net_dActivationFunction(net_zActivations[jActivation]);

		net_biasGradient[jBias] -= a * b * learnRate;

		double x = net_activations[jActivation];

		double w = 0;

		for (int j = 0; j < nIn; j++) {

			w += net_weights[jkWeight];

			net_weightGradient[jkWeight] -= a * b * x * learnRate;

			jkWeight--;

		};

		jActivation--;
		jBias--;

	};
	
	return 1;
};


void net_ApplyGradient(
	net_dataType* net_weights, net_dataType* net_bias,
	net_dataType* net_weightGradient, net_dataType* net_biasGradient,
	int nWeights, int nBias) {

	for (int b = 0; b < nWeights; b++) {
		net_weights[b] += net_weightGradient[b];
	};

	for (int w = 0; w < nBias; w++) {
		net_bias[w] += net_biasGradient[w];
	};

	return;

};
