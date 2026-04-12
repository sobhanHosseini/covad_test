from models.model_backbones import BackboneModel, MLP, End2EndModel

def joint_model(num_attr, expand_dim, freeze_parameters, use_relu = False, use_sigmoid = False, backbone = "resnet18", model_state_dict = None, mode = "train", concept_intervention = False):
    model_1 = BackboneModel(num_attr=num_attr, num_classes = 1, freeze_parameters=freeze_parameters, expand_dim=expand_dim, bottleneck=True, backbone = backbone)
    if mode == "train" and model_state_dict is not None:
        filtered_dict = {k: v for k, v in model_state_dict.items() if not k.startswith('fc_layers.0')}
        model_1.load_state_dict(filtered_dict)
        if freeze_parameters:
            print("Using frozen parameters...")
        else:
            print("Fine-tuning last layer of teacher model...")

    print(f"Using {backbone} for concept extraction (no fusion)...")

    model_2 = MLP(input_dim=num_attr, expand_dim=expand_dim)
    full_model = End2EndModel(model_1, model_2, use_relu, use_sigmoid)

    if mode == "test":
        if not concept_intervention:
            full_model.load_state_dict(model_state_dict)
        else:
            concept_state_dict = {k.replace("first_model.", ""): v
                                for k, v in model_state_dict.items()
                                if k.startswith("first_model.")
                                }
            model_1.load_state_dict(concept_state_dict, strict = False)
            main_state_dict = {k.replace("second_model.", ""): v
                                for k, v in model_state_dict.items()
                                if k.startswith("second_model.")
                                }
            model_2.load_state_dict(main_state_dict, strict = False)

    if concept_intervention:
        return model_1, model_2
    else:
        return full_model


def standard_model(freeze_parameters, backbone = "resnet18", model_state_dict = None, num_classes = 1, mode = "train"): #for label prediction without using attributes
    model = BackboneModel(num_attr=None, num_classes=num_classes, freeze_parameters=freeze_parameters, backbone = backbone)
    print(f"Using {backbone} for prediction...")
    if model_state_dict is not None:
        if mode == "train":
            print(f"Using weights of pre-trained model for attribute prediction...")
            filtered_dict = {k: v for k, v in model_state_dict.items() if not k.startswith('fc_layers.0.fc')}
            model.load_state_dict(filtered_dict)
            if freeze_parameters:
                print("Using frozen parameters...")
            else:
                print("Fine-tuning last layer...")
        if mode == "test":
            model.load_state_dict(model_state_dict)
    return model

def concepts_model(num_attr, freeze_parameters, expand_dim, backbone = "resnet18", model_state_dict = None, mode = "train"): #for concept prediction in independent and sequential model
    model = BackboneModel(num_attr=num_attr, num_classes=1, freeze_parameters=freeze_parameters, expand_dim=expand_dim, bottleneck=True, backbone=backbone)
    if mode == "train" and model_state_dict is not None:
        filtered_dict = {k: v for k, v in model_state_dict.items() if not k.startswith('fc_layers.0')}
        model.load_state_dict(filtered_dict)
        if freeze_parameters:
            print("Using frozen parameters...")
        else:
            print("Fine-tuning last layer of teacher model...")

    print(f"Using {backbone} for concept extraction (no fusion)...")

    if mode == "test":
        model.load_state_dict(model_state_dict, strict = False)

    return model

def main_model(num_attr, expand_dim, model_state_dict = None): #for main prediction in independent and sequential model
    model = MLP(input_dim=num_attr, expand_dim=expand_dim)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict) 
    return model
