from transformers import BartConfig, PretrainedConfig
class MultiModalBlenderbotConfig(PretrainedConfig):

    model_type = "blenderbot"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=8008,
        max_position_embeddings=256,
        encoder_layers=2,
        encoder_ffn_dim=10240,
        encoder_attention_heads=32,
        decoder_layers=24,
        decoder_ffn_dim=10240,
        decoder_attention_heads=32,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=2560,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=1,
        scale_embedding=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        encoder_no_repeat_ngram_size=3,
        forced_eos_token_id=2,
        **kwargs,
    ):
        self.activation_dropout = 0.0
        self.activation_function = "gelu"
        self.add_bias_logits = False
        self.add_final_layer_norm = True
        self.architectures = ["BlenderbotForConditionalGeneration"]
        self.attention_dropout = 0.0
        self.bos_token_id = 1
        self.classif_dropout = 0.0
        self.classifier_dropout = 0.0
        self.d_model = 2560
        self.decoder_attention_heads = 32
        self.decoder_ffn_dim = 10240
        self.decoder_layerdrop = 0.0
        self.decoder_layers = 12
        self.decoder_start_token_id = 1
        self.do_blenderbot_90_layernorm = True
        self.dropout = 0.1
        self.encoder_attention_heads = 32
        self.encoder_ffn_dim = 10240
        self.encoder_layerdrop = 0.0
        self.encoder_layers = 2
        self.encoder_no_repeat_ngram_size = 3
        self.eos_token_id = 2
        self.extra_layer_norm = False
        self.extra_pos_embeddings = 0
        self.force_bos_token_to_be_generated = False
        self.forced_eos_token_id = 2
        self.id2label = {
            "0": "LABEL_0",
            "1": "LABEL_1",
            "2": "LABEL_2"
          }

        self.init_std = 0.02
        self.is_encoder_decoder = True
        self.label2id ={
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2
        }
        self.layernorm_variant = "prelayernorm"
        self.length_penalty = 0.65
        self.max_length = 60
        self.max_position_embeddings = 256
        self.min_length = 10
        self.model_type = "blenderbot"
        self.no_repeat_ngram_size = 3
        self.normalize_before = True
        self.normalize_embedding = False
        self.num_beams = 10
        self.num_hidden_layers = 2
        self.pad_token_id = 0
        self.scale_embedding = True
        self.static_position_embeddings = False
        self.transformers_version = "4.24.0"
        self.unk_token_id = 3
        self.use_cache = True
        self.vocab_size = 8008

        self.pretrained_bart_model = "facebook/blenderbot-1B-distill"
        self.pretrained_blender_model = "facebook/blenderbot-1B-distill"
        self.img_bos_token = "<img>"
        self.img_eos_token = "</img>"
        self.eos_token = '</s>'
        self.mlm_token = "<mlm>"
        self.cid_token = "<cid>"
        self.mrm_token = "<mrm>"
        self.gcd_token = "<gcd>"
        self.sty_token = "<sty>"
        self.feat_token = "<feat>"
        self.zero_token = "<zero>"

        self.img_bos_token_id = None
        self.img_eos_token_id =None
        self.sty_token_id = None
        self.mrm_token_id = None
        self.cid_token_id = None
        self.gcd_token_id = None
        self.mlm_token_id = None
        self.feat_token_id = None
        self.zero_token_id = None

        self.image_feature_size = 512
        self.image_path = "../datasets/images/yfcc_images/"
        self.task_list = ['mlm', 'cid', 'mrm', 'gcd']
        self.finetuning_task = "turn_one"
        self.pretrain_weight = "../checkpoints/turn/epoch=54-step=13750train_loss=2.9367-train_perplexity=18.8533-valid_loss=3.3612-valid_perplexity=28.9832-rougeL_f=0-bleu_score=0.ckpt"
        self.max_length = 125


        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
class MultiModalBartConfig(BartConfig):
    def __init__(
        self,
        _name_or_path= "bart-base",
        activation_dropout= 0.1,
        activation_function= "gelu",
        add_bias_logits= False,
        add_final_layer_norm= False,
        architectures=["BartModel"],
        attention_dropout= 0.1,
        bos_token_id= 0,
        classif_dropout= 0.1,
        classifier_dropout= 0.0,
        d_model= 768,
        decoder_attention_heads= 12,
        decoder_ffn_dim= 3072,
        decoder_layerdrop= 0.0,
        decoder_layers= 6,
        decoder_start_token_id= 2,
        dropout= 0.1,
        early_stopping= True,
        encoder_attention_heads= 12,
        encoder_ffn_dim= 3072,
        encoder_layerdrop= 0.0,
        encoder_layers= 6,
        eos_token_id= 2,
        forced_bos_token_id= 0,
        forced_eos_token_id= 2,
        gradient_checkpointing= False,
        init_std= 0.02,
        is_encoder_decoder= True,
        max_position_embeddings= 1024,
        model_type= "bart",
        no_repeat_ngram_size= 3,
        normalize_before= False,
        normalize_embedding= True,
        num_beams= 4,
        num_hidden_layers= 6,
        pad_token_id= 1,
        scale_embedding= False,
        torch_dtype= "float32",
        transformers_version= "4.26.0",
        use_cache= True,
        vocab_size= 50265,
        pretrained_bart_model = "facebook/bart-base",
        img_bos_token = "<img>",
        img_eos_token = "</img>",
        utr_token = "<utr>",
        cap_token = "<cap>",
        mcm_token = "<mcm>", # Masked Caption Modeling
        cid_token = "<cid>", # Change Image Data
        cdd_token = "<cdd>", # Change Dialogue Data
        mlm_token = "<mlm>", # Masked Langue Modeling
        sty_token = "<sty>",
        img_bos_token_id = None,
        img_eos_token_id = None,
        cap_token_id = None,
        utr_token_id = None,
        sty_token_id=None,
        image_feature_size = 512,
        image_path = "../datasets/images/yfcc_images/",
        task_list = ['mcm', 'cid', 'cdd', 'mlm'],
        finetuning_task = "turn_one",
        pretrain_weight = "../checkpoints/turn/epoch=54-step=13750train_loss=2.9367-train_perplexity=18.8533-valid_loss=3.3612-valid_perplexity=28.9832-rougeL_f=0-bleu_score=0.ckpt",
        max_length = 125,
        **kwargs,
    ):
        super().__init__(
            _name_or_path=_name_or_path,
            activation_dropout=activation_dropout,
            activation_function=activation_function,
            add_bias_logits=add_bias_logits,
            add_final_layer_norm=add_final_layer_norm,
            architectures=architectures,
            attention_dropout=attention_dropout,
            bos_token_id=bos_token_id,
            classif_dropout=classif_dropout,
            classifier_dropout=classifier_dropout,
            d_model=d_model,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_layerdrop=decoder_layerdrop,
            decoder_layers=decoder_layers,
            decoder_start_token_id=decoder_start_token_id,
            dropout=dropout,
            early_stopping=early_stopping,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layerdrop=encoder_layerdrop,
            encoder_layers=encoder_layers,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            gradient_checkpointing=gradient_checkpointing,
            init_std=init_std,
            is_encoder_decoder=is_encoder_decoder,
            max_position_embeddings=max_position_embeddings,
            model_type=model_type,
            no_repeat_ngram_size=no_repeat_ngram_size,
            normalize_before=normalize_before,
            normalize_embedding=normalize_embedding,
            num_beams=num_beams,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            scale_embedding=scale_embedding,
            torch_dtype=torch_dtype,
            transformers_version=transformers_version,
            use_cache=use_cache,
            vocab_size=vocab_size,
            **kwargs,
        )

        self.pretrained_bart_model = pretrained_bart_model
        self.img_bos_token = img_bos_token
        self.img_eos_token = img_eos_token
        self.utr_token = utr_token
        self.cap_token = cap_token
        self.eos_token = '</s>'
        self.mcm_token = mcm_token
        self.cid_token = cid_token
        self.cdd_token = cdd_token
        self.mlm_token = mlm_token
        self.sty_token = sty_token

        self.img_bos_token_id = img_bos_token_id
        self.img_eos_token_id = img_eos_token_id
        self.cap_token_id = cap_token_id
        self.utr_token_id = utr_token_id
        self.mcm_token_id = None
        self.cid_token_id = None
        self.cdd_token_id = None
        self.mlm_token_id = None
        self.sty_token_id = None

        self.eos_token_id = 2

        self.image_feature_size = image_feature_size
        self.image_path = image_path
        self.pretrain_weight = pretrain_weight
        self.max_length = max_length

        self.task_list = task_list
        self.finetuning_task = finetuning_task