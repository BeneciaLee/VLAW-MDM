import os
import json
import argparse
import gc

import pytorch_lightning as pl

from tqdm import tqdm
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping, ModelCheckpoint

from src.model.config import MultiModalBlenderbotConfig
from src.model.modeling_blenderbot import *
from src.model.model_blender_bot import *
from src.model.dataset import *


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
        default=16,
        type=int,
        help=' '
    )

    parser.add_argument(
        '--max_epochs',
        default=20,
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
    config.pretrain_weight = ""
    config.max_position_embeddings = 256

    return config
def pretrain(args):
    config = get_config("facebook/blenderbot-400M-distill")
    config.max_epochs = args.max_epochs
    config.batch_size = args.batch_size

    with open(args.train_path) as file:
        datas = json.load(file)

    with open(args.caption_json_file) as file:
        caption_datas = json.load(file)

    captions = []
    image_hashes = []
    dialogues = []
    styles = []
    turns = []

    turns_numbers = {
        1:0,
        2:0,
        3:0
    }

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

    config.steps_per_epoch = (len(image_hashes) // config.batch_size) + 1 if ( len(image_hashes) // config.batch_size) == 0 else (len(image_hashes) // config.batch_size) + 1

    del datas, caption_datas
    gc.collect()

    print("Total number of captions : ", len(captions), "Total image hash count : ", len(image_hashes),
          "Total number of conversations : ", len(dialogues))
    if len(captions) != len(image_hashes) or len(captions) != len(dialogues) or len(image_hashes) != len(dialogues):
        raise ValueError("Unequal number of data.")


    #####################################
    for d in dialogues:
        turns_numbers[len(d)] += 1

    print(f"Turn 1 : {turns_numbers[1]} | Turn 2 : {turns_numbers[2]} | Turn 3 : {turns_numbers[3]}")

    # CLIP
    image_feautres = torch.load("../datasets/tensor_pactched_with_turn_one.pt")
    test_image_feautres = torch.load("../datasets/tensor_pactched_test.pt")

    print("image_feautres.shape", image_feautres.shape)

    model = MultiModalBlendPretraining(config)
    print("model success")
    model.to("cuda")

    vlp_dialogue_dataset = PreFinetuningDataset(config, captions, image_feautres, styles, turns)
    print("dataset success")
    vlp_dialogue_dataloader = DataLoader(vlp_dialogue_dataset, batch_size=args.batch_size,
                                         collate_fn=PreGenerateDataForPretraining(config, model.model.tokenizer), shuffle=True, num_workers=24)
    print("dataloader success")

    dirpath = f"../checkpoints/ablation/clip/mlm_cid_mrm_gcd/"
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        save_last=False,
        save_top_k=1,
        # filename='{epoch}-{step}-{train_iso_loss:.4f}-{train_iso_f1}-{train_iso_acc}',
        filename='{epoch}-{step}{mlm_loss:.2f}{mlm_perplexity:.2f}{mrm_loss:.2f}{cid_loss:.2f}{cid_perplexity:.4f}{gcd_loss:.2f}{gcd_perplexity:.2f}',
        verbose=True,
        monitor='mlm_loss',
        mode='min',
        save_on_train_epoch_end=True
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", precision=16,
                         accumulate_grad_batches=(2024//config.batch_size), amp_backend="native",
                         callbacks=[checkpoint_callback])

    trainer.fit(model, vlp_dialogue_dataloader)

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
    pretrain(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
