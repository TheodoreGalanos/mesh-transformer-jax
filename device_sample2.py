import argparse
import json
import time
from itertools import product
import jax
import numpy as np
import optax
import os
from tqdm import tqdm

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    #seq = params["seq"]
    seq = 784
    norm = params["norm"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint {ckpt_step}")

    #total_batch = per_replica_batch * jax.device_count() // cores_per_replica
    total_batch = 8
    
    prompts = [
        "[prompt] a house with five rooms [layout]",
        "[prompt] a house with six rooms [layout]",
        "[prompt] a house with seven rooms [layout]",
        "[prompt] a house with eight rooms [layout]",
        "[prompt] a house with nine rooms [layout]",
        "[prompt] a house with ten rooms [layout]",
        "[prompt] a house with one bedroom and one bathroom [layout]",
        "[prompt] a house with one bedroom and two bathrooms [layout]",
        "[prompt] a house with two bedrooms and one bathroom [layout]",
        "[prompt] a house with two bedrooms and two bathrooms [layout]",
        "[prompt] a house with two bedrooms and three bathrooms [layout]",
        "[prompt] a house with three bedrooms and one bathroom [layout]",
        "[prompt] a house with three bedrooms and two bathrooms [layout]",
        "[prompt] a house with three bedrooms and three bathrooms [layout]",
        "[prompt] a house with four bedrooms and one bathroom [layout]",
        "[prompt] a house with four bedrooms and two bathrooms [layout]",
        "[prompt] a house with four bedrooms and three bathrooms [layout]",
        "[prompt] a house with four bedrooms and four bathrooms [layout]",
        "[prompt] the bedroom is adjacent to the living room [layout]",
        "[prompt] a bedroom is adjacent to the living room [layout]",
        "[prompt] the bedroom is adjacent to the kitchen [layout]",
        "[prompt] a bedroom is adjacent to the kitchen [layout]",
        "[prompt] the bedroom is adjacent to the kitchen [layout]",
        "[prompt] the kitchen is adjacent to the bathroom [layout]",
        "[prompt] a bathroom is adjacent to the living room [layout]",
        "[prompt] the bathroom is adjacent to the living room [layout]",
        "[prompt] the bedroom is not adjacent to the living room [layout]",
        "[prompt] a bedroom is not adjacent to the living room [layout]",
        "[prompt] the bedroom is not adjacent to the kitchen [layout]",
        "[prompt] a bedroom is not adjacent to the kitchen [layout]",
        "[prompt] the bedroom is not adjacent to the kitchen [layout]",
        "[prompt] the kitchen is not adjacent to the bathroom [layout]",
        "[prompt] a bathroom is not adjacent to the living room [layout]",
        "[prompt] the bathroom is not adjacent to the living room [layout]",
        "[prompt] the bedroom is in the north side of the house [layout]",
        "[prompt] the bedroom is in the north east side of the house [layout]",
        "[prompt] the bedroom is in the east side of the house [layout]",
        "[prompt] the bedroom is in the south east side of the house [layout]",
        "[prompt] the bedroom is in the south side of the house [layout]",
        "[prompt] the bedroom is in the south west side of the house [layout]",
        "[prompt] the bedroom is in the west side of the house [layout]",
        "[prompt] the bedroom is in the north west side of the house [layout]",
        "[prompt] a bedroom is in the north side of the house [layout]",
        "[prompt] a bedroom is in the north east side of the house [layout]",
        "[prompt] a bedroom is in the east side of the house [layout]",
        "[prompt] a bedroom is in the south east side of the house [layout]",
        "[prompt] a bedroom is in the south side of the house [layout]",
        "[prompt] a bedroom is in the south west side of the house [layout]",
        "[prompt] a bedroom is in the west side of the house [layout]",
        "[prompt] a bedroom is in the north west side of the house [layout]",
        "[prompt] the kitchen is in the north side of the house [layout]",
        "[prompt] the kitchen is in the north east side of the house [layout]",
        "[prompt] the kitchen is in the east side of the house [layout]",
        "[prompt] the kitchen is in the south east side of the house [layout]",
        "[prompt] the kitchen is in the south side of the house [layout]",
        "[prompt] the kitchen is in the south west side of the house [layout]",
        "[prompt] the kitchen is in the west side of the house [layout]",
        "[prompt] the kitchen is in the north west side of the house [layout]",
    ]
    
    top_p = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]
    top_k = [0, 50, 100]
    generation_params = list(product(top_p, top_k))
    
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        
        for param_set in tqdm(generation_params):
            
            folder = 'models/generated_data/text/GPTJ/scaling_laws/{}/topP_{}_topK_{}'.format(str(args.config, param_set[0]), str(param_set[1]))
            os.makedirs(folder, exist_ok=True)
            for prompt in prompts:
                outputs = []
                for i in range(0, 25):
                    tokens = tokenizer.encode(prompt)
                    start = time.time()
                    provided_ctx = len(tokens)
                    pad_amount = seq - provided_ctx
                    #pad_amount = seq + provided_ctx
                    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
                    batched_tokens = np.array([padded_tokens] * total_batch)
                    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)
                    output = network.generate(batched_tokens, length, 512, {"top_p": np.ones(total_batch) * param_set[0],
                                                                            "top_k": np.ones(total_batch) * param_set[1],
                                                                            "temp": np.ones(total_batch) * 0.75})
                    decoded_output = []
                    for idx, o in enumerate(output[1][0][:, :, 0]):
                        decoded_output.append(tokenizer.decode(o))

                    outputs.append(decoded_output)
                flat_outputs = [item for sublist in outputs for item in sublist]
                with open(folder + '/{}.txt'.format(prompt.replace(' ', '_')), 'w', encoding='utf8') as f:
                    for output in flat_outputs:
                        f.write(output + "\n")
