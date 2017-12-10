#pragma once

#include <cassert>
#include <cmath>

#include "common.h"
#include "tensor.h"

namespace nnp {

class LinearActivation {
public:
	template <typename Float, size_t INPUT_C, size_t BATCH_SIZE>
	static Tensor<Float, INPUT_C, BATCH_SIZE> forward(Tensor<Float, INPUT_C, BATCH_SIZE> input) {
		return input;
	}

	template <typename Float, size_t INPUT_C, size_t BATCH_SIZE>
	static Tensor<Float, INPUT_C, BATCH_SIZE> backward(const Tensor<Float, INPUT_C, BATCH_SIZE>&
		, Tensor<Float, INPUT_C, BATCH_SIZE> gradient) {
		return gradient;
	}
};

class SigmoidActivation {
public:
	template <typename Float, size_t INPUT_C, size_t BATCH_SIZE>
	static Tensor<Float, INPUT_C, BATCH_SIZE> forward(Tensor<Float, INPUT_C, BATCH_SIZE> input) {
		for (auto& ii : input)
			ii = sigmoid(ii);
		return input;
	}

	template <typename Float, size_t INPUT_C, size_t BATCH_SIZE>
	static Tensor<Float, INPUT_C, BATCH_SIZE> backward(const Tensor<Float, INPUT_C, BATCH_SIZE>& sigmoid
		, Tensor<Float, INPUT_C, BATCH_SIZE> gradient) {
		assert(sigmoid.size() == gradient.size()
			&& sigmoid.batchSize() == gradient.batchSize());
		auto sigmoidIt = sigmoid.begin();
		for (auto& gg : gradient) {
			gg *= *sigmoidIt * (Float{1} - *sigmoidIt);
			++sigmoidIt;
		}
		return gradient;
	}

private:
	template <typename Float>
	static Float sigmoid(Float f) {
		return Float{1} / (Float{1} + std::exp(-f));
	}
};

} // namespace nnp
