#pragma once

#include <dlib/matrix/matrix.h>
#include <dlib/matrix/matrix_utilities.h>

#include "activation.h"
#include "common.h"
#include "tensor.h"

namespace nnp {

namespace details {

template <typename Float = float, size_t NODE_C = RESIZEABLE, size_t INPUT_C = RESIZEABLE>
class LayerWeights
{
public:
	template <typename Generator>
	explicit LayerWeights(Generator&& gen)
	{
		for (auto& w : m_weights)
			w = gen();
	}

	template <
		typename InputFloat,
		size_t BATCH_SIZE = RESIZEABLE,
		// Requirement from dlib. Matrix types have to be the same.
		typename = std::enable_if_t<std::is_same<Float, InputFloat>::value>>
	Tensor<Float, NODE_C, BATCH_SIZE>
		forward(const Tensor<InputFloat, INPUT_C, BATCH_SIZE>& input) const
	{
		return typename Tensor<Float, NODE_C, BATCH_SIZE>::Data(m_weights * input.data());
	}

	template <
		typename GradFloat,
		size_t BATCH_SIZE = RESIZEABLE,
		// Requirement from dlib. Matrix types have to be the same.
		typename = std::enable_if_t<std::is_same<Float, GradFloat>::value>>
	Tensor<Float, INPUT_C, BATCH_SIZE>
		backward(const Tensor<GradFloat, NODE_C, BATCH_SIZE>& gradient) const
	{
		return typename Tensor<Float, INPUT_C, BATCH_SIZE>::Data(
			trans(m_weights) * gradient.data());
	}

	template <
		typename InputFloat,
		typename GradFloat,
		size_t BATCH_SIZE = RESIZEABLE,
		// Requirement from dlib. Matrix types have to be the same.
		typename = std::enable_if_t<
			std::is_same<Float, GradFloat>::value && std::is_same<Float, GradFloat>::value>>
	void update(
		const Tensor<InputFloat, INPUT_C, BATCH_SIZE>& input,
		const Tensor<GradFloat, NODE_C, BATCH_SIZE>& gradient,
		Float stepSize,
		Float regularization)
	{
		m_weights -=
			stepSize * (gradient.data() * trans(input.data()) + regularization * m_weights);
	}

	Float l2Norm() const
	{
		Float sum{0};
		for (Float w : m_weights)
			sum += w * w;
		return sum;
	}

	static constexpr size_t nodeCount() { return NODE_C; }

	static constexpr size_t inputCount() { return INPUT_C; }

private:
	dlib::matrix<Float, NODE_C, INPUT_C> m_weights;
};

template <typename Float = float, size_t NODE_C = RESIZEABLE, size_t INPUT_C = RESIZEABLE>
class BiasedLayerWeights
{
public:
	template <typename Generator>
	explicit BiasedLayerWeights(Generator&& gen)
		: m_weights(gen)
	{
		for (auto& b : m_bias)
			b = 0;
	}

	template <
		typename InputFloat,
		size_t BATCH_SIZE = RESIZEABLE,
		// Requirement from dlib. Matrix types have to be the same.
		typename = std::enable_if_t<std::is_same<Float, InputFloat>::value>>
	Tensor<Float, NODE_C, BATCH_SIZE>
		forward(const Tensor<InputFloat, INPUT_C, BATCH_SIZE>& input) const
	{
		auto ret = m_weights.forward(input);
		for (size_t ii = 0; ii != ret.batchSize(); ++ii)
			for (size_t jj = 0; jj != ret.size(); ++jj)
				ret(jj, ii) += m_bias(jj);
		return ret;
	}

	template <
		typename GradFloat,
		size_t BATCH_SIZE = RESIZEABLE,
		// Requirement from dlib. Matrix types have to be the same.
		typename = std::enable_if_t<std::is_same<Float, GradFloat>::value>>
	Tensor<Float, INPUT_C, BATCH_SIZE>
		backward(const Tensor<GradFloat, NODE_C, BATCH_SIZE>& gradient) const
	{
		return m_weights.backward(gradient);
	}

	template <
		typename InputFloat,
		typename GradFloat,
		size_t BATCH_SIZE = RESIZEABLE,
		// Requirement from dlib. Matrix types have to be the same.
		typename = std::enable_if_t<
			std::is_same<Float, GradFloat>::value && std::is_same<Float, GradFloat>::value>>
	void update(
		const Tensor<InputFloat, INPUT_C, BATCH_SIZE>& input,
		const Tensor<GradFloat, NODE_C, BATCH_SIZE>& gradient,
		Float stepSize,
		Float regularization)
	{
		m_weights.update(input, gradient, stepSize, regularization);
		for (size_t jj = 0; jj != gradient.size(); ++jj)
		{
			GradFloat sum{0};
			for (size_t ii = 0; ii != gradient.batchSize(); ++ii)
				sum += gradient(jj, ii);
			m_bias(jj) -= stepSize * sum;
		}
	}

	Float l2Norm() const { return m_weights.l2Norm(); }

	static constexpr size_t nodeCount() { return NODE_C; }

	static constexpr size_t inputCount() { return INPUT_C; }

private:
	LayerWeights<Float, NODE_C, INPUT_C> m_weights;
	dlib::matrix<Float, NODE_C, 1> m_bias;
};

} // namespace details

template <
	typename Activation,
	typename Float = float,
	size_t NODE_C = RESIZEABLE,
	size_t INPUT_C = RESIZEABLE>
class ComputationalLayer
{
	using Weights = details::BiasedLayerWeights<Float, NODE_C, INPUT_C>;

public:
	template <typename Generator>
	explicit ComputationalLayer(Generator&& gen)
		: m_weights(gen) {}

	template <typename InputFloat, size_t BATCH_SIZE = RESIZEABLE>
	Tensor<Float, NODE_C, BATCH_SIZE>
		forward(const Tensor<InputFloat, INPUT_C, BATCH_SIZE>& input) const
	{
		return m_activation.forward(m_weights.forward(input));
	}

	template <typename GradFloat, size_t BATCH_SIZE = RESIZEABLE>
	Tensor<Float, INPUT_C, BATCH_SIZE> backward(
		const Tensor<GradFloat, NODE_C, BATCH_SIZE>& output,
		const Tensor<GradFloat, NODE_C, BATCH_SIZE>& gradient) const
	{
		return m_weights.backward(m_activation.backward(output, gradient));
	}

	template <typename InputFloat, typename GradFloat, size_t BATCH_SIZE = RESIZEABLE>
	void update(
		const Tensor<InputFloat, INPUT_C, BATCH_SIZE>& input,
		const Tensor<GradFloat, NODE_C, BATCH_SIZE>& gradient,
		Float stepSize,
		Float regularization)
	{
		m_weights.update(input, gradient, stepSize, regularization);
	}

	Float l2Norm() const { return m_weights.l2Norm(); }

	static constexpr size_t nodeCount() { return Weights::nodeCount(); }

	static constexpr size_t inputCount() { return Weights::inputCount(); }

private:
	Activation m_activation;
	Weights m_weights;
};

template <typename Float = float, size_t NODE_C = RESIZEABLE, size_t INPUT_C = RESIZEABLE>
using LinearLayer = ComputationalLayer<LinearActivation, Float, NODE_C, INPUT_C>;

template <typename Float = float, size_t NODE_C = RESIZEABLE, size_t INPUT_C = RESIZEABLE>
using ReluLayer = ComputationalLayer<ReluActivation, Float, NODE_C, INPUT_C>;

template <typename Float = float, size_t NODE_C = RESIZEABLE, size_t INPUT_C = RESIZEABLE>
using SigmoidLayer = ComputationalLayer<SigmoidActivation, Float, NODE_C, INPUT_C>;

} // namespace nnp
