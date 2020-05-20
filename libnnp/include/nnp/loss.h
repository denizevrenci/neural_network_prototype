#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "details/misc.h"
#include "tensor.h"

namespace nnp {

template <typename Float, size_t SIZE, size_t BATCH_SIZE>
Tensor<Float, SIZE, BATCH_SIZE> softmax(Tensor<Float, SIZE, BATCH_SIZE> input)
{
	for (size_t ii = 0; ii != input.batchSize(); ++ii)
	{
		auto max = *std::max_element(&input(0, ii), &(input(0, ii + 1)));
		for (size_t jj = 0; jj != input.size(); ++jj)
			input(jj, ii) = std::exp(input(jj, ii) - max);
		auto sum = std::accumulate(&input(0, ii), &(input(0, ii + 1)), Float{0});
		for (size_t jj = 0; jj != input.size(); ++jj)
			input(jj, ii) /= sum;
	}
	return input;
}

template <
	typename Float,
	size_t SIZE_E,
	size_t BATCH_SIZE_E,
	size_t SIZE_G,
	size_t BATCH_SIZE_G>
Float crossEntropy(
	const Tensor<Float, SIZE_E, BATCH_SIZE_E>& estimation,
	const Tensor<Float, SIZE_G, BATCH_SIZE_G>& groundTruth)
{
	assert(
		estimation.size() == groundTruth.size() &&
		estimation.batchSize() == groundTruth.batchSize());
	Float sum{0};
	for (size_t ii = 0; ii != estimation.batchSize(); ++ii)
	{
		for (size_t jj = 0; jj != estimation.size(); ++jj)
			sum -= groundTruth(jj, ii) * std::log(estimation(jj, ii));
	}
	return sum / estimation.batchSize();
}

template <typename Float>
class SoftMaxLayer
{
public:
	template <typename InputFloat, size_t SIZE, size_t BATCH_SIZE>
	static Tensor<Float, SIZE, BATCH_SIZE>
		probs(const Tensor<InputFloat, SIZE, BATCH_SIZE>& input)
	{
		return softmax(input);
	}

	template <typename PFloat, typename GFloat, size_t SIZE, size_t BATCH_SIZE>
	static Float loss(
		const Tensor<PFloat, SIZE, BATCH_SIZE>& probs,
		const Tensor<GFloat, SIZE, BATCH_SIZE>& groundTruth,
		Float totalL2Norm,
		Float regularization)
	{
		return crossEntropy(probs, groundTruth) + 0.5 * regularization * totalL2Norm;
	}

	template <typename PFloat, typename GFloat, size_t SIZE, size_t BATCH_SIZE>
	static Tensor<Float, SIZE, BATCH_SIZE> getGradient(
		Tensor<PFloat, SIZE, BATCH_SIZE> probs,
		const Tensor<GFloat, SIZE, BATCH_SIZE>& groundTruth)
	{
		for (size_t ii = 0; ii != probs.batchSize(); ++ii)
		{
			probs(details::argmax(&groundTruth(0, ii), &(groundTruth(0, ii + 1))), ii) -=
				Float{1};
			for (size_t jj = 0; jj != probs.size(); ++jj)
				probs(jj, ii) /= probs.batchSize();
		}
		return probs;
	}
};

} // namespace nnp
