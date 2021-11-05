from synthesizer.hparams_2 import hparams
from synthesizer.train_2 import train
from utils.argutils import print_args
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("run_id", type=str, help= \
    #     "Name for this model instance. If a model state from the same run ID was previously "
    #     "saved, the training will restart from there. Pass -f to overwrite saved states and "
    #     "restart from scratch.")
    # parser.add_argument("syn_dir", type=str, default=argparse.SUPPRESS, help= \
    #     "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
    #     "the wavs and the embeds.")
    # parser.add_argument("-m", "--models_dir", type=str, default="synthesizer/saved_models/", help=\
    #     "Path to the output directory that will contain the saved model weights and the logs.")
    # parser.add_argument("-s", "--save_every", type=int, default=1000, help= \
    #     "Number of steps between updates of the model on the disk. Set to 0 to never save the "
    #     "model.")
    # parser.add_argument("-b", "--backup_every", type=int, default=25000, help= \
    #     "Number of steps between backups of the model. Set to 0 to never make backups of the "
    #     "model.")
    # parser.add_argument("-f", "--force_restart", action="store_true", help= \
    #     "Do not load any saved model and restart from scratch.")
    # parser.add_argument("--hparams", default="",
    #                     help="Hyperparameter overrides as a comma-separated list of name=value "
	# 						 "pairs")

    # different for tacotron2
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    print_args(args, parser)

    args.hparams = hparams.parse(args.hparams)

    # Run the training
    train(**vars(args))
