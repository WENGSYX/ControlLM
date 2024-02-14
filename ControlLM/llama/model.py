def add_vector_after_position(matrix, vector, position_ids, after=None):
    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1
    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)
    matrix += mask.float().to(matrix.device) * vector.to(matrix.device)
    return matrix


def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i: i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    end_pos = find_subtensor_position(tokens, end_str)
    return end_pos + len(end_str) - 1


class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[:, -1, :]
            dot_product = torch.mm(last_token_activations, self.calc_dot_product_with)
            self.dot_products = dot_product

        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output + self.add_activations.to(augmented_output.device),) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class LlamaControlLM:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).half().cuda()
        self.device = self.model.device

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.model.model.layers:
            layer.after_position = pos

    def prompt_to_tokens(self, instruction):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.system_prompt = 'You are a helpful, honest and concise assistant.'
        dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()
        dialog_tokens = self.tokenizer.encode(
            f"{B_INST} {dialog_content.strip()} {E_INST}"
        )
        return torch.tensor(dialog_tokens).unsqueeze(0)

    def generate_text(self, prompt, max_new_tokens=50):
        tokens = self.prompt_to_tokens(prompt).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate(model, text, max_length=512, max_new_tokens=None):

        tokenizer = model.tokenizer
        tokenizer.pad_token = '[PAD]'
        stop_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        device = model.device
        input_ids = []
        for t in text:
            input_ids.append(t)
        min_prompt_len = min(len(t) for t in input_ids)
        max_prompt_len = max(len(t) for t in input_ids)

        if max_new_tokens:
            max_length = max_prompt_len + max_new_tokens
        tokens = torch.full((len(input_ids), max_length), pad_id, dtype=torch.long).to(device)
        for k, t in enumerate(input_ids):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        prev_pos = 0
        cur_pos = min_prompt_len - 1
        input_text_mask = tokens != pad_id
        eos_reached = torch.tensor([False] * len(input_ids), device=device)
        past_key_values = None

        with torch.no_grad():
            for cur_pos_add in range(max_length):
                cur_pos += 1
                if prev_pos != 0:
                    prev_pos = cur_pos - 1
                if tokens.shape[1] == cur_pos:
                    break
                torch.cuda.empty_cache()

                logits = model.model(tokens[:, prev_pos:cur_pos], use_cache=True,
                                     past_key_values=past_key_values)
                next_token = torch.topk(logits['logits'][:, -1], 1, dim=-1)[1][:, -1]
                next_token = next_token.reshape(-1)
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
                eos_reached |= (~input_text_mask[:, cur_pos]) & (
                        next_token == model.tokenizer.eos_token_id
                )

                if all(eos_reached):
                    break
                prev_pos = cur_pos
                past_key_values = logits["past_key_values"]
        return tokens

    def __call__(self, input_ids):
        with torch.no_grad():
            logits = self.model(input_ids)
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        activations = activations.half()
        self.model.model.layers[layer].add(activations)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def get_and_save_activations(self, dataset, save_path):
        os.makedirs(save_path)
        system_prompt = "You are a helpful, honest and concise assistant."

        def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
            dialog_tokens = tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
            )
            return torch.tensor(dialog_tokens).unsqueeze(0)

        def generate_and_save_steering_vectors(
                model, dataset, file, start_layer=0, end_layer=79, token_idx=-2
        ):
            layers = list(range(start_layer, end_layer + 1))
            positive_activations = dict([(layer, []) for layer in layers])
            negative_activations = dict([(layer, []) for layer in layers])
            model.set_save_internal_decodings(False)
            model.reset_all()
            for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
                p_tokens = p_tokens.to(model.device)
                n_tokens = n_tokens.to(model.device)
                model.reset_all()
                model(p_tokens)
                for layer in layers:
                    p_activations = model.get_last_activations(layer)
                    p_activations = p_activations[0, -1, :].detach().cpu()
                    positive_activations[layer].append(p_activations)
                model.reset_all()
                model(n_tokens)
                for layer in layers:
                    n_activations = model.get_last_activations(layer)
                    n_activations = n_activations[0, -1, :].detach().cpu()
                    negative_activations[layer].append(n_activations)
            for layer in layers:
                positive = torch.stack(positive_activations[layer])
                negative = torch.stack(negative_activations[layer])
                vec = (positive - negative).mean(dim=0)
                torch.save(vec, save_path + f"/vec_layer_{layer}.pt")
                torch.save(positive, save_path + f"/positive_layer_{layer}.pt")
                torch.save(
                    negative,
                    save_path + f"/negative_layer_{layer}.pt",
                )

        class ComparisonDataset(Dataset):
            def __init__(self, data, system_prompt):
                self.data = data
                self.system_prompt = system_prompt
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                question = item["question"]
                pos_answer = item["answer_matching_behavior"]
                neg_answer = item["answer_not_matching_behavior"]
                pos_tokens = prompt_to_tokens(
                    self.tokenizer, self.system_prompt, question, pos_answer
                )
                neg_tokens = prompt_to_tokens(
                    self.tokenizer, self.system_prompt, question, neg_answer
                )
                return pos_tokens, neg_tokens

        data = ComparisonDataset(dataset, system_prompt)
        generate_and_save_steering_vectors(self, data, save_path, end_layer=len(self.model.model.layers) - 1)
        print('Now, We have sueecss save the activate in {}'.format(save_path))

    def load_and_set_activate(self, load_path=None, layer=20, gamma=1):
        self.reset_all()
        control_activate = torch.load(os.path.join(load_path, 'vec_layer_{}.pt'.format(layer)))
        if gamma is not None:
            if type(gamma) in [int, float]:
                control_activate = gamma * control_activate
        self.set_add_activations(layer, control_activate)
