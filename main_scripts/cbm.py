import argparse
import os
import torch
import pandas as pd
import numpy as np
import gc

from torch.utils.data import ConcatDataset
from utils.model_utils import generate_concept_logits
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, standard_model, concepts_model, main_model
from trainers.trainer_cbm import CBMTrainer
from evaluators.evaluator_cbm import CBMEvaluator

def load_dataset(df, split, use_attr = True, load_image = True, multiclass = False, anomaly_ratio = 1.0, contaminate = False, n_per_type = 0, subsample_anomalies = False, original_df = None, random_state = 42):
    return ConceptDataset(df, split=split, use_attr=use_attr, load_image=load_image, multiclass=multiclass, random_state=random_state, contaminate=contaminate, subsample_anomalies = subsample_anomalies, n_per_type=n_per_type, original_df=original_df)

def make_dataloader(dataset, batch_size, shuffle = True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(category: str, 
                dataframe_path: str,
                dataframe_path_original: str,
                subsample_anomalies: bool,
                contaminate: bool, #add some original images to the training set of generated anomalies
                n_per_type: int, #number of original images per defect to add
                anomaly_ratio: float,
                model_type: str, 
                save_dir: str,
                device: torch.device, 
                backbone: str,
                expand_dim: int,
                lambda_: float,
                batch_size: int = 16, 
                optimizer: str = "adam", 
                lr: float = 1e-3, 
                epochs: int = 100, 
                use_concepts: bool = True, 
                multiclass: bool = False,
                freeze_parameters: bool = False, 
                model_path: str = None,
                save_path_new_df: str = None,
                use_gen_anomalies: bool = False,
                seed: int = 42):
    
    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr = lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "min", factor = 0.1, patience = 5)
        return opt, scheduler

    #base directory
    if use_gen_anomalies and contaminate:
        sub_dir = f"gen_anomalies_weakly_sup_{n_per_type}"
    elif use_gen_anomalies and not contaminate:
        sub_dir = "gen_anomalies"
    elif not use_gen_anomalies and subsample_anomalies:
        sub_dir = f"original_anomalies_weakly_sup_{n_per_type}"
    elif not use_gen_anomalies and not subsample_anomalies:
        sub_dir = "original_anomalies"
    save_dir = os.path.join(save_dir, sub_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)

    #handle subfolders for certain models
    if model_type in ["sequential", "independent"]:
        os.makedirs(os.path.join(save_dir, "main"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "concepts"), exist_ok=True)

    #build save paths
    if model_type in ["joint", "standard"]:
        save_path = os.path.join(save_dir, f"{backbone}_{anomaly_ratio}ratio_{expand_dim}MLP_automated.pth")
    elif model_type in ["sequential", "independent"]:
        save_path = os.path.join(save_dir, "main", f"{backbone}_{anomaly_ratio}ratio_{expand_dim}MLP_automated.pth")
        save_path_concepts = os.path.join(save_dir, "concepts", f"{backbone}_{anomaly_ratio}ratio_{expand_dim}MLP_automated.pth")
    
    print(f"\nTraining {model_type} model for {category} category...")

    dataframe = pd.read_csv(dataframe_path)
    dataframe_original = pd.read_csv(dataframe_path_original) if contaminate else None

    state_dict = torch.load(model_path) if model_path else None

    train_dataset = load_dataset(dataframe, "train", use_attr=use_concepts, multiclass=multiclass, contaminate = contaminate, n_per_type=n_per_type, subsample_anomalies=subsample_anomalies, original_df = dataframe_original, random_state=seed)
    val_dataset = load_dataset(dataframe, "val", use_attr=use_concepts, multiclass=multiclass, random_state=seed)

    train_dataloader = make_dataloader(train_dataset, batch_size)
    val_dataloader = make_dataloader(val_dataset, batch_size, shuffle = False)

    num_classes = train_dataset.num_classes if multiclass else None
    if multiclass:
        print(f"Number of labels: {num_classes}")
    num_attr = len(train_dataset.attr_cols) if use_concepts else None

    if model_type == "independent":
        train_dataset_no_img = load_dataset(dataframe, "train", load_image=False, contaminate=contaminate, n_per_type=n_per_type, subsample_anomalies=subsample_anomalies, original_df=dataframe_original, random_state=seed)
        val_dataset_no_img = load_dataset(dataframe, "val", load_image=False, random_state=seed)

        train_dataloader_no_img = make_dataloader(train_dataset_no_img, batch_size)
        val_dataloader_no_img = make_dataloader(val_dataset_no_img, batch_size)

    print(f"\nNumber of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    if not multiclass:
        imbalance_ratio, contamination_ratio = train_dataset.find_class_imbalance("main") 
        print("Imbalance Ratio (negatives per positive) in training set:", imbalance_ratio)
        print("Contamination ratio in training set:", contamination_ratio)
    else:
        imbalance_ratio = None

    imbalance_ratio_attr = None
    if use_concepts:
        imbalance_ratio_attr = train_dataset.find_class_imbalance("attributes")

    if model_type == "joint":
        model = joint_model(num_attr=num_attr, expand_dim=expand_dim, use_relu = True, use_sigmoid=False, 
                            freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone)
        # save train and val dataframe in the state dict
        model.train_df = pd.concat([train_dataset.df, val_dataset.df], ignore_index=True)
        model.to(device)
        optimizer, scheduler = init_optimizer(model.parameters())
        trainer = CBMTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, scheduler, 
                                device, lambda_ = lambda_, num_epochs=epochs, concepts=use_concepts, 
                                weight_main=imbalance_ratio, weight_attr=imbalance_ratio_attr, save_path=save_path) 
        
        val_main_f1, val_attr_f1 = trainer.train()

    elif model_type == "standard":
        if multiclass:
            model = standard_model(freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone, num_classes=num_classes)
        else:
            model = standard_model(freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone, num_classes=1)
        model.to(device)
        optimizer, scheduler = init_optimizer(model.parameters())
        trainer = CBMTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, scheduler,
                                device, lambda_ = lambda_, num_epochs=epochs, concepts=False, 
                                main_only=True, multiclass=multiclass, weight_main=imbalance_ratio, save_path=save_path) 
       
        val_main_f1 = trainer.train()

    elif model_type == "independent" or model_type == "sequential":
        # common first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=freeze_parameters, 
                                       expand_dim=expand_dim, model_state_dict=state_dict, backbone=backbone)
        concept_model.to(device)
        concept_optimizer, concept_scheduler = init_optimizer(concept_model.parameters())

        trainer_concepts = CBMTrainer(concept_model, num_attr, train_dataloader, val_dataloader, concept_optimizer, concept_scheduler,
                                         device, lambda_ = lambda_, num_epochs=epochs, concepts=True, 
                                         bottleneck=True, weight_attr=imbalance_ratio_attr, save_path=save_path_concepts)

        val_attr_f1 = trainer_concepts.train()

        #main model: common
        main_task_model = main_model(num_attr=num_attr, expand_dim = 8)
        main_task_model.to(device)
        main_optimizer, main_scheduler = init_optimizer(main_task_model.parameters())

        if model_type == "sequential":
            #generate concept logits
            dataframe_new = generate_concept_logits(concept_model, dataframe, save_path = save_path_new_df, device = device)

            train_dataset_no_img = load_dataset(dataframe_new, "train", load_image=False, contaminate=contaminate, n_per_type=n_per_type, subsample_anomalies=subsample_anomalies, original_df=dataframe_original, random_state=seed)
            val_dataset_no_img = load_dataset(dataframe_new, "val", load_image=False, random_state=seed)

            train_dataloader_no_img = make_dataloader(train_dataset_no_img, batch_size)
            val_dataloader_no_img = make_dataloader(val_dataset_no_img, batch_size)

        trainer_main = CBMTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img, 
                                    main_optimizer, main_scheduler, device, lambda_ = lambda_, num_epochs=epochs, 
                                    concepts=False, main_only=True, weight_main=imbalance_ratio, save_path=save_path)

        val_main_f1 = trainer_main.train()

    torch.cuda.empty_cache()
    gc.collect()


def test_model(category: str,
               dataframe_path: str,
               subsample_anomalies: bool, 
               contaminate: bool,
               n_per_type: int,               
               anomaly_ratio: float,
               model_type: str, 
               save_dir: str,
               device: torch.device, 
               backbone: str,
               expand_dim: int,
               batch_size: int = 16, 
               use_concepts: bool = True,
               mode: str = "eval",
               image_path: str = None,
               use_gen_anomalies: bool = False):

    #base directory
    if use_gen_anomalies and contaminate:
        sub_dir = f"gen_anomalies_weakly_sup_{n_per_type}"
    elif use_gen_anomalies and not contaminate:
        sub_dir = "gen_anomalies"
    elif not use_gen_anomalies and subsample_anomalies:
        sub_dir = f"original_anomalies_weakly_sup_{n_per_type}"
    elif not use_gen_anomalies and not subsample_anomalies:
        sub_dir = "original_anomalies"
    save_dir = os.path.join(save_dir, sub_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)

    #handle subfolders for certain models
    if model_type in ["sequential", "independent"]:
        os.makedirs(os.path.join(save_dir, "main"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "concepts"), exist_ok=True)

    save_path_concepts = None
    #build save paths
    if model_type in ["joint", "standard"]:
        save_path = os.path.join(save_dir, f"{backbone}_{anomaly_ratio}ratio_{expand_dim}MLP_automated.pth")
    elif model_type in ["sequential", "independent"]:
        save_path = os.path.join(save_dir, "main", f"{backbone}_{anomaly_ratio}ratio_{expand_dim}MLP_automated.pth")
        save_path_concepts = os.path.join(save_dir, "concepts", f"{backbone}_{anomaly_ratio}ratio_{expand_dim}MLP_automated.pth")
    
    print(f"\nTesting {model_type} model for {category} category...")

    dataframe = pd.read_csv(dataframe_path)

    print(f"Loading state dict from {save_path}")

    state_dict = torch.load(save_path, weights_only=False) if save_path else None
    state_dict_concepts = torch.load(save_path_concepts) if save_path_concepts else None

    test_dataset = load_dataset(dataframe, "test", use_attr=use_concepts)
    num_attr = len(test_dataset.attr_cols) if use_concepts else None
    attr_cols = test_dataset.attr_cols

    if use_gen_anomalies:
        val_dataset = load_dataset(dataframe, "val", use_attr=use_concepts)
        train_dataset = load_dataset(dataframe, "train", use_attr=use_concepts)
        test_df = pd.concat([train_dataset.df, val_dataset.df, test_dataset.df], ignore_index=True)

        print(f"test_df len: {len(test_df)}")
        print(f"val_df len: {len(val_dataset)}")
        print(f"train_df len: {len(train_dataset)}")

        if state_dict is not None:
            train_df = state_dict.get("train_df", None)
            # access only the train and val splits
            if train_df is not None:
                # remove from train_df the test split
                train_df = train_df[train_df["split"].isin(["train", "val"])]
                # to check occurrences, we can use the image paths, but only the filename and dirname of the parent folder and not the full path
                def get_rel_path(path):
                    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
                train_df["rel_path"] = train_df["image_path"].apply(get_rel_path)
                # find original anomalies which are the ones that don't have "gen" in the filepath
                train_df = train_df[~train_df["image_path"].str.contains("gen")]

                test_df["rel_path"] = test_df["image_path"].apply(get_rel_path)

                # remove rows that occur both in train_df and dataframe (test set)
                test_df = test_df[~test_df["rel_path"].isin(train_df["rel_path"])]
                test_df = test_df.drop(columns=["rel_path"])

        # repeated code
        test_dataset = load_dataset(test_df, "test", use_attr=use_concepts)
        num_attr = len(test_dataset.attr_cols) if use_concepts else None
        attr_cols = test_dataset.attr_cols
        val_dataset = load_dataset(test_df, "val", use_attr=use_concepts)
        train_dataset = load_dataset(test_df, "train", use_attr=use_concepts)
        test_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    test_dataloader = make_dataloader(test_dataset, batch_size, shuffle = False)

    #test_dataset = load_dataset(dataframe, "test", use_attr=use_concepts)

    test_dataloader = make_dataloader(test_dataset, batch_size, shuffle = False)


    if mode == "eval":
        print(f"\nNumber of test images: {len(test_dataset)}")

    elif mode == "inference":
        print(f"\nPerforming inference on image stored in {image_path}...")

    if model_type == "joint":
        model = joint_model(num_attr=num_attr, expand_dim=expand_dim, use_relu = True, use_sigmoid=False, 
                                freeze_parameters=True, model_state_dict=state_dict, backbone=backbone, mode = "test")
        
        model.to(device)

        evaluator = CBMEvaluator(model, num_attr, attr_cols, test_dataloader, device, concepts=use_concepts)

        if mode == "eval":
            auc_main, f1_main, auc_attr, f1_attr, image_preds = evaluator.evaluate()
        elif mode == "inference":
            evaluator.inference(image_path)
    
    elif model_type == "standard":
        model = standard_model(freeze_parameters=True, model_state_dict=state_dict, backbone=backbone, mode="test")
        model.to(device)
        evaluator = CBMEvaluator(model, num_attr, attr_cols, test_dataloader, device, concepts=False, main_only=True)

        if mode == "eval":
            auc_main, f1_main = evaluator.evaluate()
        elif mode == "inference":
            evaluator.inference(image_path)
    
    elif model_type in ["independent", "sequential"]:
        #first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=True, 
                                    expand_dim=expand_dim, model_state_dict=state_dict_concepts, backbone=backbone, mode = "test")
        concept_model.to(device)

        concept_evaluator = CBMEvaluator(concept_model, num_attr, attr_cols, test_dataloader, device, concepts = True, bottleneck = True)

        if mode == "eval":
            auc_attr, f1_attr = concept_evaluator.evaluate()
        elif mode == "inference":
            concept_evaluator.inference(image_path)

        #second step: main prediction
        main_task_model = main_model(num_attr=num_attr, expand_dim = 8, model_state_dict=state_dict)
        main_task_model.to(device)

        #generate concept logits
        dataframe_new = generate_concept_logits(concept_model, dataframe, splits=["test"], save_path = None, device = device)
        test_dataset_no_img = load_dataset(dataframe_new, "test", load_image=False)
        test_dataloader_no_img = make_dataloader(test_dataset_no_img, batch_size)

        main_evaluator = CBMEvaluator(main_task_model, num_attr, attr_cols, test_dataloader_no_img, device, main_only = True)

        if mode == "eval":
            auc_main, f1_main = main_evaluator.evaluate()
        
        elif mode == "inference":
            main_evaluator.inference(image_path)

    if mode == "eval":
        return auc_main, auc_attr, f1_main, f1_attr 
    else:
        return None, None, None, None

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, help="Whether to train, test or perform inference on one image")
    parser.add_argument("--dataframe_path", type=str, help="Path to dataframe")
    parser.add_argument("--dataframe_path_original", type=str, help="Path to original dataframe to add original images")
    parser.add_argument("--subsample_anomalies", action="store_true", help="Whether to use a reduced number of anomalous images per defect type")
    parser.add_argument("--contaminate", action="store_true", help="Whether to contaminate the generated dataset with some original anomalous samples")
    parser.add_argument("--n_per_type", type=int, default = 0, help="How many original images to add to the generated dataset")
    parser.add_argument("--anomaly_ratio", type=int, default = 1.0, help="Anomaly ratios to keep in training")
    parser.add_argument("--model_type", type = str, nargs="+", help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--save_dir", type=str, help="Directory to save the models in")
    parser.add_argument("--category", type = str, help="Which category to train/test")
    parser.add_argument("--device", type=str, default=None, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--expand_dim", type=int, default = 0, help="How many neurons to use in FC layers")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch size to use")
    parser.add_argument("--lambda_", type=float, default = 0.55, help="How much weight to give to the attributes")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--use_concepts", action="store_true", help="Whether to use concepts or not")
    parser.add_argument("--multiclass", action="store_true", help="Train the model to perform multiclass classification")
    parser.add_argument("--freeze_parameters", action="store_true", help="Whether to freeze the parameters of the network for concept prediction")
    parser.add_argument("--model_path", type=str, default = None, help="If specified, loads the state dict of a chosen model")
    parser.add_argument("--save_concepts", action="store_true", help="Whether to save the predicted concepts dataframe")
    parser.add_argument("--seeds", type=int, nargs="+", default=42, help="Execution seed")
    parser.add_argument("--image_path", default=None, help="Path of the image to perform inference on")
    parser.add_argument("--use_gen_anomalies", action="store_true", help="Perform training on dataset with generated anomalies")
    parser.add_argument("--gemini_logo_mask_path", default=None, help="Path to the Gemini logo mask to be applied to all images")
    
    args = parser.parse_args()

    # set the gemini_logo_mask_path as an environment variable
    if args.gemini_logo_mask_path is not None:
        os.environ["GEMINI_LOGO_MASK_PATH"] = args.gemini_logo_mask_path
    else:
        os.environ.pop("GEMINI_LOGO_MASK_PATH", None)

    torch.manual_seed(args.seeds)
    resolved_device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(resolved_device)
    print(f"Using device: {device}")

    for model_type in args.model_type:           
        if args.mode == "train":
            train_model(args.category, args.dataframe_path, args.dataframe_path_original, args.subsample_anomalies, args.contaminate, args.n_per_type, args.anomaly_ratio, model_type, args.save_dir, device, args.backbone, args.expand_dim, args.lambda_, args.batch_size, args.optimizer, args.lr, args.epochs, args.use_concepts, args.multiclass, args.freeze_parameters, args.model_path, args.save_concepts, args.use_gen_anomalies, args.seeds)
        elif args.mode == "eval" or args.mode == "inference":
            test_auc_main, test_auc_attr, test_f1_main, test_f1_attr = test_model(args.category, args.dataframe_path, args.subsample_anomalies, args.contaminate, args.n_per_type, args.anomaly_ratio, model_type, args.save_dir, device, args.backbone, args.expand_dim, args.batch_size, args.use_concepts, args.mode, args.image_path, args.use_gen_anomalies)


if __name__ == "__main__":
    main()