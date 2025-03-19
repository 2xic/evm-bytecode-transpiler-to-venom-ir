# https://etherscan.io/address/0xeefba1e63905ef1d7acba5a8513c70307c1ce441#contracts
MULTICALL = bytes.fromhex("608060405234801561001057600080fd5b50600436106100885760003560e01c806372425d9d1161005b57806372425d9d146100e757806386d516e8146100ef578063a8b0574e146100f7578063ee82ac5e1461010c57610088565b80630f28c97d1461008d578063252dba42146100ab57806327e86d6e146100cc5780634d2301cc146100d4575b600080fd5b61009561011f565b6040516100a2919061051e565b60405180910390f35b6100be6100b93660046103b6565b610123565b6040516100a292919061052c565b610095610231565b6100956100e2366004610390565b61023a565b610095610247565b61009561024b565b6100ff61024f565b6040516100a2919061050a565b61009561011a3660046103eb565b610253565b4290565b60006060439150825160405190808252806020026020018201604052801561015f57816020015b606081526020019060019003908161014a5790505b50905060005b835181101561022b576000606085838151811061017e57fe5b6020026020010151600001516001600160a01b031686848151811061019f57fe5b6020026020010151602001516040516101b891906104fe565b6000604051808303816000865af19150503d80600081146101f5576040519150601f19603f3d011682016040523d82523d6000602084013e6101fa565b606091505b50915091508161020957600080fd5b8084848151811061021657fe5b60209081029190910101525050600101610165565b50915091565b60001943014090565b6001600160a01b03163190565b4490565b4590565b4190565b4090565b600061026382356105d4565b9392505050565b600082601f83011261027b57600080fd5b813561028e61028982610573565b61054c565b81815260209384019390925082018360005b838110156102cc57813586016102b68882610325565b84525060209283019291909101906001016102a0565b5050505092915050565b600082601f8301126102e757600080fd5b81356102f561028982610594565b9150808252602083016020830185838301111561031157600080fd5b61031c8382846105ee565b50505092915050565b60006040828403121561033757600080fd5b610341604061054c565b9050600061034f8484610257565b825250602082013567ffffffffffffffff81111561036c57600080fd5b610378848285016102d6565b60208301525092915050565b600061026382356105df565b6000602082840312156103a257600080fd5b60006103ae8484610257565b949350505050565b6000602082840312156103c857600080fd5b813567ffffffffffffffff8111156103df57600080fd5b6103ae8482850161026a565b6000602082840312156103fd57600080fd5b60006103ae8484610384565b60006102638383610497565b61041e816105d4565b82525050565b600061042f826105c2565b61043981856105c6565b93508360208202850161044b856105bc565b60005b84811015610482578383038852610466838351610409565b9250610471826105bc565b60209890980197915060010161044e565b50909695505050505050565b61041e816105df565b60006104a2826105c2565b6104ac81856105c6565b93506104bc8185602086016105fa565b6104c58161062a565b9093019392505050565b60006104da826105c2565b6104e481856105cf565b93506104f48185602086016105fa565b9290920192915050565b600061026382846104cf565b602081016105188284610415565b92915050565b60208101610518828461048e565b6040810161053a828561048e565b81810360208301526103ae8184610424565b60405181810167ffffffffffffffff8111828210171561056b57600080fd5b604052919050565b600067ffffffffffffffff82111561058a57600080fd5b5060209081020190565b600067ffffffffffffffff8211156105ab57600080fd5b506020601f91909101601f19160190565b60200190565b5190565b90815260200190565b919050565b6000610518826105e2565b90565b6001600160a01b031690565b82818337506000910152565b60005b838110156106155781810151838201526020016105fd565b83811115610624576000848401525b50505050565b601f01601f19169056fea265627a7a72305820978cd44d5ce226bebdf172bdf24918753b9e111e3803cb6249d3ca2860b7a47f6c6578706572696d656e74616cf50037")

MINIMAL_PROXY = bytes.fromhex("363d3d373d3d3d363d73c04bd2f0d484b7e0156b21c98b2923ca8b9ce1495af43d82803e903d91602b57fd5bf3")
MINIMAL_PROXY_2 = bytes.fromhex("36602c57343d527f9e4ac34f21c619cefc926c8bd93b54bf5a39c7ab2127a895af1cc0691d7e3dff593da1005b363d3d373d3d3d3d6100e6806062363936013d7381a89a3c934b688f93368b07d1210c7d669e27bc5af43d3d93803e606057fd5bf3c02aaa39b223fe8d0a0e5c4f27ead9083c756cc287c22db324b8b0637c8f09d2670ae7777651dbb8a5033a6bdb31e52ce6ba9c67bff7331ac2686e72e4f48c5e50e21c6437511bd60b26b8c69e52122400000000000000000000000000000000000000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff006a94d74f4300000000000000000000002386f26fc100000d2f13f7789f00000065c3a9000065c79d800000000000000000000000470de4df82000001000000000000000000000000000000000000000000000000000000000000000000e6")

ERC4626_RRATE_PROVIDER = bytes.fromhex("608060405234801561001057600080fd5b50600436106100415760003560e01c806313511b5c14610046578063679aefce1461008a57806371f29123146100a0575b600080fd5b61006d7f000000000000000000000000ad55aebc9b8c03fc43cd9f62260391c13c23e7c081565b6040516001600160a01b0390911681526020015b60405180910390f35b6100926100c7565b604051908152602001610081565b6100927f0000000000000000000000000000000000000000000000000de0b6b3a764000081565b6040516303d1689d60e11b81527f0000000000000000000000000000000000000000000000000de0b6b3a764000060048201526000907f000000000000000000000000ad55aebc9b8c03fc43cd9f62260391c13c23e7c06001600160a01b0316906307a2d13a90602401602060405180830381865afa15801561014e573d6000803e3d6000fd5b505050506040513d601f19601f820116820180604052508101906101729190610177565b905090565b60006020828403121561018957600080fd5b505191905056fea264697066735822122063bfd17fb86af2bf43c40db0ad396342fae7d50aea4d5f8ceb14aaff5e8e67ac64736f6c63430008180033")
REGISTRY = bytes.fromhex("608080604052600436101561001357600080fd5b600090813560e01c9081630e2009f414610216575080633b2bcbf1146101a7578063698eec4414610138578063b7010697146100c95763c57981b51461005857600080fd5b346100c657807ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc3601126100c657602060405173ffffffffffffffffffffffffffffffffffffffff7f000000000000000000000000368457ec2c9096b9ae9e30af1b20aa16ce422930168152f35b80fd5b50346100c657807ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc3601126100c657602060405173ffffffffffffffffffffffffffffffffffffffff7f0000000000000000000000001a5b13ef713ba23f7bb17013d1c73797e1285f26168152f35b50346100c657807ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc3601126100c657602060405173ffffffffffffffffffffffffffffffffffffffff7f000000000000000000000000a6c9976893eacb13688c16689ea183ce29004855168152f35b50346100c657807ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc3601126100c657602060405173ffffffffffffffffffffffffffffffffffffffff7f000000000000000000000000d85ee50da419cc5af83a1e70a91d5c630b8c650a168152f35b90503461028457817ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc3601126102845760209073ffffffffffffffffffffffffffffffffffffffff7f000000000000000000000000ee705735c5d02c24d738b0f7dd30a4abd741a160168152f35b5080fdfea26469706673582212205c59516c768bbb139dfb0dc1b778a63f67a5bb728c0a0c949564f72be8b42c9664736f6c63430008130033")
ERC721_DROP = bytes.fromhex("60806040523661001357610011610017565b005b6100115b610027610022610074565b6100b9565b565b606061004e83836040518060600160405280602781526020016102fb602791396100dd565b9392505050565b73ffffffffffffffffffffffffffffffffffffffff163b151590565b90565b60006100b47f360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc5473ffffffffffffffffffffffffffffffffffffffff1690565b905090565b3660008037600080366000845af43d6000803e8080156100d8573d6000f35b3d6000fd5b606073ffffffffffffffffffffffffffffffffffffffff84163b610188576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602660248201527f416464726573733a2064656c65676174652063616c6c20746f206e6f6e2d636f60448201527f6e7472616374000000000000000000000000000000000000000000000000000060648201526084015b60405180910390fd5b6000808573ffffffffffffffffffffffffffffffffffffffff16856040516101b0919061028d565b600060405180830381855af49150503d80600081146101eb576040519150601f19603f3d011682016040523d82523d6000602084013e6101f0565b606091505b509150915061020082828661020a565b9695505050505050565b6060831561021957508161004e565b8251156102295782518084602001fd5b816040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161017f91906102a9565b60005b83811015610278578181015183820152602001610260565b83811115610287576000848401525b50505050565b6000825161029f81846020870161025d565b9190910192915050565b60208152600082518060208401526102c881604085016020870161025d565b601f017fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe016919091016040019291505056fe416464726573733a206c6f772d6c6576656c2064656c65676174652063616c6c206661696c6564a2646970667358221220b76547697fda83e5f0b4980957aff7124665dcb9ff2d4c85d579d3afef1c94ea64736f6c634300080a0033")

# Failing atm
RESTACKING_POOL = bytes.fromhex("608060405261000c61000e565b005b7f0000000000000000000000006ab15b49ad9cb743a403850fad9e09aaa12c8f5c6001600160a01b0316330361007b576000356001600160e01b03191663278f794360e11b14610071576040516334ad5dbb60e21b815260040160405180910390fd5b610079610083565b565b6100796100b2565b6000806100933660048184610312565b8101906100a09190610352565b915091506100ae82826100c2565b5050565b6100796100bd61011d565b610155565b6100cb82610179565b6040516001600160a01b038316907fbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b90600090a28051156101155761011082826101f5565b505050565b6100ae61026b565b60006101507f360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc546001600160a01b031690565b905090565b3660008037600080366000845af43d6000803e808015610174573d6000f35b3d6000fd5b806001600160a01b03163b6000036101b457604051634c9c8ce360e01b81526001600160a01b03821660048201526024015b60405180910390fd5b7f360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc80546001600160a01b0319166001600160a01b0392909216919091179055565b6060600080846001600160a01b0316846040516102129190610422565b600060405180830381855af49150503d806000811461024d576040519150601f19603f3d011682016040523d82523d6000602084013e610252565b606091505b509150915061026285838361028a565b95945050505050565b34156100795760405163b398979f60e01b815260040160405180910390fd5b60608261029f5761029a826102e9565b6102e2565b81511580156102b657506001600160a01b0384163b155b156102df57604051639996b31560e01b81526001600160a01b03851660048201526024016101ab565b50805b9392505050565b8051156102f95780518082602001fd5b604051630a12f52160e11b815260040160405180910390fd5b6000808585111561032257600080fd5b8386111561032f57600080fd5b5050820193919092039150565b634e487b7160e01b600052604160045260246000fd5b6000806040838503121561036557600080fd5b82356001600160a01b038116811461037c57600080fd5b9150602083013567ffffffffffffffff8082111561039957600080fd5b818501915085601f8301126103ad57600080fd5b8135818111156103bf576103bf61033c565b604051601f8201601f19908116603f011681019083821181831017156103e7576103e761033c565b8160405282815288602084870101111561040057600080fd5b8260208601602083013760006020848301015280955050505050509250929050565b6000825160005b818110156104435760208186018101518583015201610429565b50600092019182525091905056fea264697066735822122027f80176059c85af4ee8751015dd9e5db167736b2eedc4be4d13ffc86fde444964736f6c63430008140033")
BULK_SENDER_ETH = bytes.fromhex("60806040526004361061004a5760003560e01c80633659cfe6146100545780634f1ef286146100875780635c60da1b146101075780638f28397014610138578063f851a4401461016b575b610052610180565b005b34801561006057600080fd5b506100526004803603602081101561007757600080fd5b50356001600160a01b031661019a565b6100526004803603604081101561009d57600080fd5b6001600160a01b0382351691908101906040810160208201356401000000008111156100c857600080fd5b8201836020820111156100da57600080fd5b803590602001918460018302840111640100000000831117156100fc57600080fd5b5090925090506101d4565b34801561011357600080fd5b5061011c610281565b604080516001600160a01b039092168252519081900360200190f35b34801561014457600080fd5b506100526004803603602081101561015b57600080fd5b50356001600160a01b03166102be565b34801561017757600080fd5b5061011c610378565b6101886103a3565b610198610193610403565b610428565b565b6101a261044c565b6001600160a01b0316336001600160a01b031614156101c9576101c481610471565b6101d1565b6101d1610180565b50565b6101dc61044c565b6001600160a01b0316336001600160a01b03161415610274576101fe83610471565b6000836001600160a01b031683836040518083838082843760405192019450600093509091505080830381855af49150503d806000811461025b576040519150601f19603f3d011682016040523d82523d6000602084013e610260565b606091505b505090508061026e57600080fd5b5061027c565b61027c610180565b505050565b600061028b61044c565b6001600160a01b0316336001600160a01b031614156102b3576102ac610403565b90506102bb565b6102bb610180565b90565b6102c661044c565b6001600160a01b0316336001600160a01b031614156101c9576001600160a01b0381166103245760405162461bcd60e51b81526004018080602001828103825260368152602001806105766036913960400191505060405180910390fd5b7f7e644d79422f17c01e4894b5f4f588d331ebfa28653d42ae832dc59e38c9798f61034d61044c565b604080516001600160a01b03928316815291841660208301528051918290030190a16101c4816104b1565b600061038261044c565b6001600160a01b0316336001600160a01b031614156102b3576102ac61044c565b6103ab61044c565b6001600160a01b0316336001600160a01b031614156103fb5760405162461bcd60e51b81526004018080602001828103825260328152602001806105446032913960400191505060405180910390fd5b610198610198565b7fe99d12b39ab17aef0ca754554afa48519dcb96ca64603696637dea37e965a6175490565b3660008037600080366000845af43d6000803e808015610447573d6000f35b3d6000fd5b7fd605002b0407d620d5ea33643507867180e600a98b93d382fc50227c2095905e5490565b61047a816104d5565b6040516001600160a01b038216907fbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b90600090a250565b7fd605002b0407d620d5ea33643507867180e600a98b93d382fc50227c2095905e55565b6104de8161053d565b6105195760405162461bcd60e51b815260040180806020018281038252603b8152602001806105ac603b913960400191505060405180910390fd5b7fe99d12b39ab17aef0ca754554afa48519dcb96ca64603696637dea37e965a61755565b3b15159056fe43616e6e6f742063616c6c2066616c6c6261636b2066756e6374696f6e2066726f6d207468652070726f78792061646d696e43616e6e6f74206368616e6765207468652061646d696e206f6620612070726f787920746f20746865207a65726f206164647265737343616e6e6f742073657420612070726f787920696d706c656d656e746174696f6e20746f2061206e6f6e2d636f6e74726163742061646472657373a265627a7a72315820ffc6ae767daeb42cea941165a56b4809d598c445576f676ae2da333a7627f49f64736f6c63430005100032")
OVE_MEV_ESCROW = bytes.fromhex("60806040526004361015610044575b3615610018575f80fd5b7f7cb3607a76b32d6d17ca5d1aeb444650b19ac0fabbb1f24c93a0da57346b56106020604051348152a1005b5f3560e01c80634641257d146100a65763fbfa77cf0361000e57346100a2575f3660031901126100a2576040517f000000000000000000000000c3cf55551058872a8b21f38514c2fb6f82ef09b86001600160a01b03168152602090f35b5f80fd5b346100a2575f3660031901126100a25760206100c06100c8565b604051908152f35b7f000000000000000000000000c3cf55551058872a8b21f38514c2fb6f82ef09b86001600160a01b0316330361018c5747908115610187577f8e55ccfc9778ff8eba1646d765cf1982537ce0f9257054a17b48aad7452501836020604051848152a1333b156100a257604051630a62ade560e31b81525f8160048186335af1801561017c576101545750565b67ffffffffffffffff811161016857604052565b634e487b7160e01b5f52604160045260245ffd5b6040513d5f823e3d90fd5b5f9150565b604051630d599dd960e11b8152600490fdfea2646970667358221220d619b98571e571c794168db3e6f87328e3cc06e55b5467159fa6ab0e831142cf64736f6c63430008160033")
