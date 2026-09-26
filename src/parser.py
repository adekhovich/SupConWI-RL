import argparse


def get_parser():

    parser = argparse.ArgumentParser(description="SupConWI-RL")
    
    # Data related
    parser.add_argument("--data_name", default="mixedwm38", type=str, help="Name of the dataset")
    parser.add_argument("--data_path", default="../../MixedDefect/data/MixedWM38.npz", type=str, help="Path of the dataset")
    parser.add_argument("--input_channels", default=1, type=int, help="Number of image channels")
    parser.add_argument("--patch_size", default=4, type=int, help="Patch size")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    
    # Classifier related:
    parser.add_argument("--classifier_name", default="lstm", type=str, help="Name of the classification model")
    parser.add_argument("--classifier_hidden_dim", default=256, type=int, help="Size of the hidden state")
    parser.add_argument("--classifier_embedding_dim", default=128, type=int, help="Size of the contrastive learning embedding")
    parser.add_argument("--classifier_num_layers", default=2, type=int, help="Number of RNN cells")
    parser.add_argument("--classifier_num_epochs", default=500, type=int, help="Number of training epochs for the classifier")
    parser.add_argument("--classifier_lr", default=1e-3, type=float, help="Learning rate for the classifier")
    parser.add_argument("--classifier_wd", default=1e-5, type=float, help="Weight decay for the classifier")
    parser.add_argument("--classifier_optimizer_name", default="Adam", type=str, help="Optimizer to train the classifier")
    parser.add_argument("--classifier_batch_size", default=128, type=int, help="Batch size to train the classifier") 
    parser.add_argument("--classifier_num_heads", default=8, type=int, help="Number of RNN heads for which the loss is computed") 
    parser.add_argument("--train_classifier", action="store_true", help="")
    parser.add_argument("--supcon", action="store_true", help="Include SupCon loss. Only for ResNet")
    
    # Confidnet related:   
    parser.add_argument("--confidnet_name", default="lstm", type=str, help="Name of the confidnet model")
    parser.add_argument("--confidnet_hidden_dim", default=128, type=int, help="Size of the hidden state")
    parser.add_argument("--confidnet_num_layers", default=2, type=int, help="Number of RNN cells")
    parser.add_argument("--confidnet_num_epochs", default=300, type=int, help="Number of training epochs for the confidnet")
    parser.add_argument("--confidnet_lr", default=1e-3, type=float, help="Learning rate for the confidnet")
    parser.add_argument("--confidnet_wd", default=1e-5, type=float, help="Weight decay for the confidnet")
    parser.add_argument("--confidnet_optimizer_name", default="Adam", type=str, help="Optimizer to train the confidnet")
    parser.add_argument("--confidnet_batch_size", default=128, type=int, help="Batch size to train the confidnet")
    parser.add_argument("--train_confidnet", action="store_true", help="")
    
    # Patch selector related:
    parser.add_argument("--patch_selector_num_iters", default=100, type=int, help="Number of training epochs for the patch_selector")
    parser.add_argument("--patch_selector_lr", default=1e-3, type=float, help="Learning rate for the patch_selector")
    parser.add_argument("--patch_selector_wd", default=1e-5, type=float, help="Weight decay for the patch_selector")
    parser.add_argument("--patch_selector_batch_size", default=16, type=int, help="Batch size to train the patch_selector")  
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor")
    parser.add_argument("--train_patch_selector", action="store_true", help="")
    
    # Hybrid & ensemble related:
    parser.add_argument("--hybrid", action="store_true", help="Hybrid approach (with CNN)")
    parser.add_argument("--aux_model_name", default="resnet34", type=str, help="Name of the auxiliary model")    
    parser.add_argument("--ensemble", action="store_true", help="Ensemble")
    parser.add_argument("--ensemble_type", default="avg", type=str, help="Type of policies aggregation")
    parser.add_argument("--num_models", default=5, type=int, help="Number of models in the ensemble. ONLY FOR EVALUATION")    
    
    return parser
    