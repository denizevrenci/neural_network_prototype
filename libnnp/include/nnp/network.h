#pragma once

#include <tuple>

#include "common.h"
#include "details/tuple.h"
#include "layer.h"

namespace nnp {

template <typename... Layers>
class TupleNetwork
{
	using LayerTuple = std::tuple<Layers...>;

	template <size_t IDX>
	using LayerType = std::tuple_element_t<IDX, LayerTuple>;

public:
	template <typename... Types>
	explicit constexpr TupleNetwork(Types&&... types)
		: m_layers(std::forward<Types>(types)...) {}

	static constexpr size_t layerCount() { return sizeof...(Layers); }

	static constexpr size_t outputCount() { return LayerType<layerCount() - 1>::nodeCount(); }

	static constexpr size_t inputCount() { return LayerType<0>::inputCount(); }

	template <typename Next, typename InputFloat, size_t BATCH_SIZE = RESIZEABLE>
	Tensor<InputFloat, inputCount(), BATCH_SIZE> propagate(
		Next&& next,
		const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input,
		InputFloat stepSize,
		InputFloat regularization)
	{
		return PropagateHelper<0>()(
			this, next, input, InputFloat{0}, stepSize, regularization);
	}

	template <typename Next, typename InputFloat, size_t BATCH_SIZE = RESIZEABLE>
	void propagate(
		Next&& next,
		const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input,
		InputFloat regularization)
	{
		PropagateHelper<0>()(this, next, input, InputFloat{0}, regularization);
	}

	template <typename InputFloat, size_t BATCH_SIZE = RESIZEABLE>
	Tensor<InputFloat, outputCount(), BATCH_SIZE>
		forward(const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input)
	{
		return ForwardHelper<layerCount() - 1>()(this, input);
	}

private:
	static_assert(layerCount() > 0, "There must be at least one layer in a TupleNetwork");

	template <size_t LAYER_IDX, typename Dummy = void> // Only partial specializations are
	                                                   // allowed in class scope.
	struct PropagateHelper
	{
		template <typename Next, typename InputFloat, size_t BATCH_SIZE>
		Tensor<InputFloat, LayerType<LAYER_IDX>::inputCount(), BATCH_SIZE> operator()(
			TupleNetwork* object,
			Next&& next,
			const Tensor<InputFloat, LayerType<LAYER_IDX>::inputCount(), BATCH_SIZE>& input,
			InputFloat totalL2Norm,
			InputFloat stepSize,
			InputFloat regularization) const
		{
			auto& thisLayer = object->getLayer<LAYER_IDX>();
			auto output = thisLayer.forward(input);
			totalL2Norm += thisLayer.l2Norm();
			auto gradient = PropagateHelper<LAYER_IDX + 1, Dummy>()(
				object, next, output, totalL2Norm, stepSize, regularization);
			auto nextGrad = thisLayer.backward(output, gradient);
			thisLayer.update(input, gradient, stepSize, regularization);
			return nextGrad;
		}

		template <typename Next, typename InputFloat, size_t BATCH_SIZE>
		void operator()(
			TupleNetwork* object,
			Next&& next,
			const Tensor<InputFloat, LayerType<LAYER_IDX>::inputCount(), BATCH_SIZE>& input,
			InputFloat totalL2Norm,
			InputFloat regularization) const
		{
			auto& thisLayer = object->getLayer<LAYER_IDX>();
			PropagateHelper<LAYER_IDX + 1, Dummy>()(
				object, next, thisLayer.forward(input), totalL2Norm, regularization);
		}
	};

	template <typename Dummy>
	struct PropagateHelper<layerCount() - 1, Dummy>
	{
		template <typename Next, typename InputFloat, size_t BATCH_SIZE>
		Tensor<InputFloat, LayerType<layerCount() - 1>::inputCount(), BATCH_SIZE> operator()(
			TupleNetwork* object,
			Next&& next,
			const Tensor<InputFloat, LayerType<layerCount() - 1>::inputCount(), BATCH_SIZE>&
				input,
			InputFloat totalL2Norm,
			InputFloat stepSize,
			InputFloat regularization) const
		{
			auto& thisLayer = object->getLayer<layerCount() - 1>();
			auto output = thisLayer.forward(input);
			totalL2Norm += thisLayer.l2Norm();
			auto gradient = next.propagate(output, totalL2Norm, stepSize, regularization);
			auto nextGrad = thisLayer.backward(output, gradient);
			thisLayer.update(input, gradient, stepSize, regularization);
			return nextGrad;
		}

		template <typename Next, typename InputFloat, size_t BATCH_SIZE>
		void operator()(
			TupleNetwork* object,
			Next&& next,
			const Tensor<InputFloat, LayerType<layerCount() - 1>::inputCount(), BATCH_SIZE>&
				input,
			InputFloat totalL2Norm,
			InputFloat regularization) const
		{
			auto& thisLayer = object->getLayer<layerCount() - 1>();
			next.propagate(thisLayer.forward(input), totalL2Norm, regularization);
		}
	};

	template <size_t LAYER_IDX, typename Dummy = void> // Only partial specializations are
													   // allowed in class scope.
	struct ForwardHelper
	{
		template <typename InputFloat, size_t BATCH_SIZE>
		Tensor<InputFloat, LayerType<LAYER_IDX>::nodeCount(), BATCH_SIZE> operator()(
			TupleNetwork* object,
			const Tensor<InputFloat, TupleNetwork::inputCount(), BATCH_SIZE>& input) const
		{
			auto& thisLayer = object->getLayer<LAYER_IDX>();
			return thisLayer.forward(ForwardHelper<LAYER_IDX - 1, Dummy>()(object, input));
		}
	};

	template <typename Dummy>
	struct ForwardHelper<0, Dummy>
	{
		template <typename InputFloat, size_t BATCH_SIZE>
		Tensor<InputFloat, LayerType<0>::nodeCount(), BATCH_SIZE> operator()(
			TupleNetwork* object,
			const Tensor<InputFloat, TupleNetwork::inputCount(), BATCH_SIZE>& input) const
		{
			auto& thisLayer = object->getLayer<0>();
			return thisLayer.forward(input);
		}
	};

	LayerTuple m_layers;

	template <size_t IDX>
	constexpr auto& getLayer()
	{
		return std::get<IDX>(m_layers);
	}

	template <size_t IDX>
	constexpr const auto& getLayer() const
	{
		return std::get<IDX>(m_layers);
	}
};

template <typename HiddenLayers, typename LossLayer>
class Network
{
	using Layers = std::tuple<HiddenLayers, LossLayer>;
	using HLayers = std::decay_t<HiddenLayers>;

public:
	template <typename... Types>
	explicit constexpr Network(Types&&... types)
		: m_layers(std::forward<Types>(types)...) {}

	static constexpr size_t layerCount() { return HLayers::layerCount(); }

	static constexpr size_t inputCount() { return HLayers::inputCount(); }

	template <typename InputFloat, typename GFloat, size_t BATCH_SIZE = RESIZEABLE>
	auto propagate(
		const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input,
		const Tensor<GFloat, HLayers::outputCount(), BATCH_SIZE>& groundTruth,
		InputFloat stepSize,
		InputFloat regularization)
	{
		LossLayerHelper<InputFloat, BATCH_SIZE> helper(lossLayer(), groundTruth);
		hiddenLayers().propagate(helper, input, stepSize, regularization);
		return helper.loss();
	}

	template <typename InputFloat, typename GFloat, size_t BATCH_SIZE = RESIZEABLE>
	auto propagate(
		const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input,
		const Tensor<GFloat, HLayers::outputCount(), BATCH_SIZE>& groundTruth,
		InputFloat regularization)
	{
		LossLayerHelper<InputFloat, BATCH_SIZE> helper(lossLayer(), groundTruth);
		hiddenLayers().propagate(helper, input, regularization);
		return helper.loss();
	}

private:
	template <typename Float, size_t BATCH_SIZE>
	class LossLayerHelper
	{
		using GroundTruth = Tensor<Float, HLayers::outputCount(), BATCH_SIZE>;

	public:
		LossLayerHelper(LossLayer& lossLayer, const GroundTruth& groundTruth)
			: m_lossLayer(&lossLayer)
			, m_groundTruth(&groundTruth) {}

		template <typename InputFloat>
		auto propagate(
			const Tensor<InputFloat, HLayers::outputCount(), BATCH_SIZE>& input,
			InputFloat totalL2Norm,
			InputFloat,
			InputFloat regularization)
		{
			auto probs = m_lossLayer->probs(input);
			m_loss = m_lossLayer->loss(probs, *m_groundTruth, totalL2Norm, regularization);
			return m_lossLayer->getGradient(probs, *m_groundTruth);
		}

		template <typename InputFloat>
		void propagate(
			const Tensor<InputFloat, HLayers::outputCount(), BATCH_SIZE>& input,
			InputFloat totalL2Norm,
			InputFloat regularization)
		{
			auto probs = m_lossLayer->probs(input);
			m_loss = m_lossLayer->loss(probs, *m_groundTruth, totalL2Norm, regularization);
		}

		Float loss() const { return m_loss; }

	private:
		LossLayer* m_lossLayer;
		const GroundTruth* m_groundTruth;
		Float m_loss;
	};

	Layers m_layers;

	constexpr auto& hiddenLayers() { return std::get<0>(m_layers); }

	constexpr const auto& hiddenLayers() const { return std::get<0>(m_layers); }

	constexpr auto& lossLayer() { return std::get<1>(m_layers); }

	constexpr const auto& lossLayer() const { return std::get<1>(m_layers); }
};

} // namespace nnp
