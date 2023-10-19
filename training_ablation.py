import os
import json
import argparse
import gc
import nltk

import pytorch_lightning as pl

from tqdm import tqdm
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

from src.model.config import MultiModalBlenderbotConfig
from src.model.modeling_blenderbot import *
from src.model.model_blender_bot import *
from src.model.dataset import *
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_path',
        default='../datasets/images/yfcc_images/',
        type=str,
        help='Images that require extraction of caption information.'
    )

    parser.add_argument(
        '--caption_json_file',
        default='../datasets/yfcc_images_added_captions.json',
        type=str,
        help='JSON file containing images for which caption information extraction is desired.'
    )

    parser.add_argument(
        '--train_path',
        default='../datasets/train_deleted_candidates.json',
        type=str,
        help=' '
    )

    parser.add_argument(
        '--test_path',
        default='../datasets/test_deleted_candidates.json',
        type=str,
        help=' '
    )

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help=' '
    )

    parser.add_argument(
        '--max_epochs',
        default=10,
        type=int,
        help=' '
    )

    #patience
    parser.add_argument(
        '--patience',
        default=15,
        type=int,
        help=' '
    )

    parser.add_argument(
        '--seed',
        default=428,
        type=int,
        help='seed'
    )

    args = parser.parse_args()
    return args

def get_config(mname = "facebook/blenderbot-400M-distill"):
    config = AutoConfig.from_pretrained(mname)

    config.pretrained_blender_model = mname
    config.img_bos_token = "<img>"
    config.img_eos_token = "</img>"
    config.eos_token = '</s>'
    config.mlm_token = "<mlm>"
    config.cid_token = "<cid>"
    config.mrm_token = "<mrm>"
    config.gcd_token = "<gcd>"
    config.sty_token = "<sty>"
    config.feat_token = "<feat>"
    config.zero_token = "<zero>"

    config.img_bos_token_id = None
    config.img_eos_token_id = None
    config.sty_token_id = None
    config.mrm_token_id = None
    config.cid_token_id = None
    config.gcd_token_id = None
    config.mlm_token_id = None
    config.feat_token_id = None
    config.zero_token_id = None
    config.truncation_max_length = 40

    config.image_feature_size = 512
    config.image_path = "../datasets/images/yfcc_images/"
    config.task_list = ['mlm', 'cid', 'mrm', 'gcd']
    config.test_turn = "turn_one"
    config.pretrain_weight = "../checkpoints/ablation/mlm_cid_mrm_gcd/pretraining.ckpt"
    config.max_position_embeddings = 256

    return config


def get_data_from_path_for_testing(config, args, task, model, image_feautres):
    with open(getattr(args, "train_path" if task == "train" else "test_path")) as file:
        datas = json.load(file)

    with open(args.caption_json_file) as file:
        caption_datas = json.load(file)

    captions = []
    image_hashes = []
    dialogues = []
    styles = []
    turns = []

    for data in tqdm(datas[:]):
        dialog = data['dialog']
        image_hash = data['image_hash']

        tmp = []
        tmp_style = []
        tmp_turn = []

        for d in dialog:
            tmp.append(" ".join(d))
            tmp_style.append(d[0])
            tmp_turn.append(d[1])

        if (image_hash + ".jpg") not in caption_datas or len(dialog) < 1:
            continue

        dialogues.append(tmp)
        styles.append(tmp_style)
        turns.append(tmp_turn)
        captions.append(caption_datas[image_hash + ".jpg"][0])
        image_hashes.append(image_hash + ".jpg")
    print("###############################")
    # Captions: ['a view of a road with a stop sign and a building in the background']
    # Image_hashes: ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
    # Dialogues: [['Appreciative (Grateful)<sty>home sweet home', 'Glamorous<sty>in my big house', 'Appreciative (Grateful)<sty>Its a house, so like it']]
    print("Captions: ", captions[:1])
    print("Image_hashes : ", image_hashes[:1])
    print("Dialogues : ", dialogues[:1])
    print("Styles : ", styles[:1])
    print("Turns : ", turns[:1])
    print("pretrain_weight : ", config.pretrain_weight)

    config.steps_per_epoch = (len(image_hashes) // config.batch_size) + 1 if (len(image_hashes) // config.batch_size) == 0 else (len(image_hashes) // config.batch_size) + 1

    del datas, caption_datas
    gc.collect()

    print("Total number of captions : ", len(captions), "Total image hash count : ", len(image_hashes), "Total number of conversations : ", len(dialogues))
    if len(captions) != len(image_hashes) or len(captions) != len(dialogues) or len(image_hashes) != len(dialogues):
        raise ValueError("Unequal number of data.")

    vlp_dialogue_dataset = None
    vlp_dialogue_dataloader = None

    vlp_dialogue_dataset = PreFinetuningDataset(config, captions, image_feautres, styles, turns)

    # mapping_test_turn = {
    #     'turn_one': 0,
    #     'turn_two': 1,
    #     'turn_three': 2
    # }
    vlp_dialogue_dataloader_turn_one = DataLoader(vlp_dialogue_dataset, batch_size=args.batch_size,
                                         collate_fn=PreGenerateDataForTest(config, model.tokenizer, test_turn = 'turn_one'),
                                         shuffle=False, num_workers=24)
    vlp_dialogue_dataloader_turn_two = DataLoader(vlp_dialogue_dataset, batch_size=args.batch_size,
                                                  collate_fn=PreGenerateDataForTest(config, model.tokenizer, test_turn='turn_two'),
                                                  shuffle=False, num_workers=24)
    vlp_dialogue_dataloader_turn_three = DataLoader(vlp_dialogue_dataset, batch_size=args.batch_size,
                                                  collate_fn=PreGenerateDataForTest(config, model.tokenizer, test_turn='turn_three'),
                                                  shuffle=False, num_workers=24)

    print("dataloader success")
    return vlp_dialogue_dataset, vlp_dialogue_dataloader_turn_one, vlp_dialogue_dataloader_turn_two, vlp_dialogue_dataloader_turn_three
def get_data_from_path(config, args, task, model, image_feautres):
    with open(getattr(args, "train_path" if task == "train" else "test_path")) as file:
        datas = json.load(file)

    with open(args.caption_json_file) as file:
        caption_datas = json.load(file)

    captions = []
    image_hashes = []
    dialogues = []
    styles = []
    turns = []

    for data in tqdm(datas[:]):
        dialog = data['dialog']
        image_hash = data['image_hash']

        tmp = []
        tmp_style = []
        tmp_turn = []

        for d in dialog:
            tmp.append(" ".join(d))
            tmp_style.append(d[0])
            tmp_turn.append(d[1])

        if (image_hash + ".jpg") not in caption_datas or len(dialog) < 1:
            continue

        dialogues.append(tmp)
        styles.append(tmp_style)
        turns.append(tmp_turn)
        captions.append(caption_datas[image_hash + ".jpg"][0])
        image_hashes.append(image_hash + ".jpg")
    print("###############################")
    # Captions: ['a view of a road with a stop sign and a building in the background']
    # Image_hashes: ['5eaa7034d31688ef1f9bed67f1f04f49.jpg']
    # Dialogues: [['Appreciative (Grateful)<sty>home sweet home', 'Glamorous<sty>in my big house', 'Appreciative (Grateful)<sty>Its a house, so like it']]
    print("Captions: ", captions[:1])
    print("Image_hashes : ", image_hashes[:1])
    print("Dialogues : ", dialogues[:1])
    print("Styles : ", styles[:1])
    print("Turns : ", turns[:1])

    del datas, caption_datas
    gc.collect()

    print("Total number of captions : ", len(captions), "Total image hash count : ", len(image_hashes), "Total number of conversations : ", len(dialogues))
    if len(captions) != len(image_hashes) or len(captions) != len(dialogues) or len(image_hashes) != len(dialogues):
        raise ValueError("Unequal number of data.")

    vlp_dialogue_dataset = None
    vlp_dialogue_dataloader = None

    vlp_dialogue_dataset = PreFinetuningDataset(config, captions, image_feautres, styles, turns)
    print("dataset success")
    if task == "train":
        vlp_dialogue_dataloader = DataLoader(vlp_dialogue_dataset, batch_size=args.batch_size,
                                         collate_fn=PreGenerateDataForFinetuning(config, model.tokenizer), shuffle=True, num_workers=24)
    else:
        vlp_dialogue_dataloader = DataLoader(vlp_dialogue_dataset, batch_size=args.batch_size,
                                             collate_fn=PreGenerateDataForValidation(config, model.tokenizer),
                                             shuffle=False, num_workers=24)

    print("dataloader success")
    return vlp_dialogue_dataset, vlp_dialogue_dataloader


from tqdm import tqdm

def training(args):
    nltk.download('punkt')
    config = get_config("facebook/blenderbot-400M-distill")
    config.max_epochs = args.max_epochs
    config.batch_size = args.batch_size

    model = MultiModalBlendFinetuning(config)
    model_path = config.pretrain_weight
    pretrain_weight = torch.load(model_path)
    model.load_state_dict(pretrain_weight['state_dict'], strict=True)

    image_feautres = torch.load("../datasets/tensor_pactched_with_turn_one.pt")
    test_image_feautres = torch.load("../datasets/tensor_pactched_test.pt")
    print("image_feautres.shape", image_feautres.shape)
    print("test_image_feautres.shape", test_image_feautres.shape)

    # config, args, task, model, image_feautres
    vlp_dialogue_dataset, vlp_dialogue_dataloader = get_data_from_path(config=config, args=args, task="train",
                                                                       model=model.model, image_feautres=image_feautres)
    vlp_dialogue_test_dataset, vlp_dialogue_dataloader_turn_one, vlp_dialogue_dataloader_turn_two, vlp_dialogue_dataloader_turn_three = get_data_from_path_for_testing(config=config, args=args, task="test",
                                                                                 model=model.model,
                                                                                 image_feautres=test_image_feautres)

    dirpath = f"../checkpoints/finetuning/clip/mlm_cid_mrm_gcd/"
    # {epoch}-{step}-{train_ccd_loss:.4f}-{train_ccd_f1}-{train_iso_loss:.4f}-{train_iso_f1}-{train_cdd_loss:.4f}-{train_cdd_f1}-{train_cod_loss:.4f}-{train_cod_f1}
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        save_last=True,
        save_top_k=1,
        filename='{epoch}-{step}{train_loss:.2f}{train_ppl:.2f}-{t1_ppl:.2f}-{t1_f1:.2f}-{t1_B:.2f}-{t1_R:.2f}-{t2_ppl:.2f}-{t2_f1:.2f}-{t2_B:.2f}-{t2_R:.2f}-{t3_ppl:.2f}-{t3_f1:.2f}-{t3_B:.2f}-{t3_R:.2f}',
        verbose=True,
        monitor='t1_f1',
        mode='max',
        save_on_train_epoch_end=False
    )

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    # trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", precision=16,
    #                      accumulate_grad_batches=(256 // config.batch_size), amp_backend="native")
    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", precision=16,
                         accumulate_grad_batches=(256//config.batch_size), amp_backend="native",
                         callbacks=[progress_bar, EarlyStopping('t1_f1', patience=args.patience, mode='max', verbose=False), checkpoint_callback])
    # trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", precision=16, amp_backend="native",
    #                      callbacks=[progress_bar,
    #                                 EarlyStopping('t1_f1', patience=args.patience, mode='max', verbose=False),
    #                                 checkpoint_callback])


    trainer.fit(model, vlp_dialogue_dataloader, [vlp_dialogue_dataloader_turn_one, vlp_dialogue_dataloader_turn_two, vlp_dialogue_dataloader_turn_three])
    # trainer.validate(model, [vlp_dialogue_dataloader_turn_two, vlp_dialogue_dataloader_turn_three])
    # trainer.fit(model, vlp_dialogue_dataloader, vlp_dialogue_dataloader_turn_one)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    seed_everything(args.seed)
    training(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
