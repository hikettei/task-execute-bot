
import argparse


parser = argparse.ArgumentParser(description='SIA Command Line Tool')
parser.add_argument('option', help='Choose from compile / run / train / help / run_all')
parser.add_argument('-k', help="SIA inflates your dataset, based on k.", default=20)
parser.add_argument('-c', '--checkpoint', help="-c <True, False>. if False, SIA skips the process of inflating your dataset.", default="True")
parser.add_argument('--ignore_json', default="False")
args = parser.parse_args()
option = args.option

if option == "compile":
    import sia_binary
    sia_binary.start_compile(args)
elif option == "train":
    import trainer
elif option == "run":
    import sia_discord
elif option == "run_all":
    import sia_binary
    sia_binary.start_compile(args)
    print("Compiling is successfully done.")
    import trainer
    print("Training is succesfully done. Starting discord bot...")
    import sia_discord
else:
    parser.print_help()