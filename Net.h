
#pragma once

#define net_dataType double

#define net_nWeights(nIn, nWidth, nDepth, nOut) ((nIn * nWidth) + ((nWidth * nWidth) * (nDepth - 1)) + (nWidth * nOut))
#define net_nBias(nWidth, nDepth, nOut) ((nWidth * nDepth) + nOut)

#define NET_INPUTS net_dataType*
#define NET_WEIGHTS net_dataType*
#define NET_BIAS net_dataType*
#define NET_ACTIVATIONS net_dataType*
#define NET_OUTPUTS net_dataType*

net_dataType* net_InitiateBlank(int length);

net_dataType* net_InitiateRandom(int length, int negativesAllowed);

void net_DisplayModel(net_dataType* net_weights, net_dataType* net_bias,
	int nIn, int nWidth, int nDepth, int nOut);

void net_DisplayActivations(net_dataType* net_inputs, net_dataType* net_activations, net_dataType* net_outputs,
	int nIn, int nWidth, int nDepth, int nOut);

void net_DisplayGradient(net_dataType* net_weightGradient, net_dataType* net_biasGradient,
	int nIn, int nWidth, int nDepth, int nOut);

double net_ActivationFunction(double activation);

double net_dActivationFunction(double activation);

int net_Infer(
	net_dataType* net_inputs, net_dataType* net_weights, net_dataType* net_bias,
	net_dataType* net_activations, net_dataType* net_zActivations, net_dataType* net_outputs,
	int nIn, int nWidth, int nDepth, int nOut);

int net_Backprop(
	net_dataType* net_inputs, net_dataType* net_weights, net_dataType* net_bias,
	net_dataType* net_activations, net_dataType* net_zActivations,
	int nIn, int nWidth, int nDepth, int nOut,
	net_dataType* net_desiredOutputs,
	net_dataType* net_weightGradient,
	net_dataType* net_biasGradient,
	double learnRate);

void net_ApplyGradient(
	net_dataType* net_weights, net_dataType* net_bias,
	net_dataType* net_weightGradient, net_dataType* net_biasGradient,
	int nWeights, int nBias);
