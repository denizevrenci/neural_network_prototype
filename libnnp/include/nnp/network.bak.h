#pragma once

#include <tuple>

#include "common.h"
#include "impl/tuple.h"
#include "layer.h"

namespace nnp {

template <typename Layer, typename NextLayer>
class SingleLayerNetwork {
public:
	SingleLayerNetwork(NextLayer& nextLayer)
		: m_nextLayer(&nextLayer) {
	}

	static constexpr size_t layerCount() { return 1; }

	static constexpr size_t inputCount() { return Layer::inputCount(); }

	static constexpr size_t outputCount() { return Layer::nodeCount(); }

	template <typename Float, size_t BATCH_SIZE = RESIZEABLE>
	Tensor<Float, outputCount(), BATCH_SIZE>
		forward(const Tensor<Float, inputCount(), BATCH_SIZE>& input) {
		return m_layer.forward(input);
	}

	/*template <typename Float, size_t BATCH_SIZE = RESIZEABLE>
	Tensor<Float, outputCount(), BATCH_SIZE>
		propagate(const Tensor<Float, inputCount(), BATCH_SIZE>& input) {
		auto output = m_layer.forward(input);

	}*/


private:
	Layer m_layer;
	NextLayer* m_nextLayer;
};

template <typename Float = float, size_t NODE_C = RESIZEABLE, size_t INPUT_C = RESIZEABLE>
using SigmoidLayerN = SingleLayerNetwork<SigmoidLayer<Float, NODE_C, INPUT_C>>;

template <typename... Subnets>
class TupleNetwork {
	using SubnetTuple = std::tuple<Subnets...>;

	template <size_t IDX>
	using SubnetType = std::tuple_element_t<IDX, SubnetTuple>;

public:
	static constexpr size_t subnetCount() { return sizeof...(Subnets); }

	static constexpr size_t layerCount() {
		LayerCountHelper helper;
		impl::forEach<SubnetTuple>(helper);
		return helper.sum;
	}

	/*constexpr auto& outSubnet() { return getSubnet<0>(); }

	constexpr const auto& outSubnet() const { return getSubnet<0>(); }

	constexpr auto& inSubnet() { return getSubnet<subnetCount() - 1>(); }

	constexpr const auto& inSubnet() const { return getSubnet<subnetCount() - 1>(); }*/

	static constexpr size_t outputCount() {
		return SubnetType<0>::outputCount();
	}

	static constexpr size_t inputCount() {
		return SubnetType<subnetCount() - 1>::inputCount();
	}

	template <typename InputFloat, size_t BATCH_SIZE = RESIZEABLE>
	Tensor<InputFloat, outputCount(), BATCH_SIZE> forward(const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input) {
		return ForwardHelper<0, InputFloat, BATCH_SIZE>()(this, input);
	}

	/*void propagate() {
		nextsubnet.forward(forward(nextsubnet.forward))
	}*/

private:
	static_assert(subnetCount() > 0, "There must be at least one subnet in an IterativeNetwork");

	struct LayerCountHelper {
		template <typename Subnet>
		constexpr void operator ()() {
			sum += Subnet::layerCount();
		}

		size_t sum = 0;
	};

	template <size_t SUBNET_IDX, typename InputFloat, size_t BATCH_SIZE>
	struct ForwardHelper {
		Tensor<InputFloat, SubnetType<SUBNET_IDX>::outputCount(), BATCH_SIZE>
			operator ()(TupleNetwork* object, const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input) {
			return object->getSubnet<SUBNET_IDX>().forward(ForwardHelper<SUBNET_IDX + 1, InputFloat, BATCH_SIZE>()(object, input));
		}
	};

	template <typename InputFloat, size_t BATCH_SIZE>
	struct ForwardHelper<subnetCount() - 1, InputFloat, BATCH_SIZE> {
		Tensor<InputFloat, SubnetType<subnetCount() - 1>::outputCount(), BATCH_SIZE>
			operator ()(TupleNetwork* object, const Tensor<InputFloat, inputCount(), BATCH_SIZE>& input) {
			return object->getSubnet<subnetCount() - 1>().forward(input);
		}
	};

	SubnetTuple m_subnets;

	template <size_t IDX>
	constexpr auto& getSubnet() { return std::get<IDX>(m_subnets); }

	template <size_t IDX>
	constexpr const auto& getSubnet() const { return std::get<IDX>(m_subnets); }
};

} // namespace nnp
