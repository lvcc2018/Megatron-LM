import os

# TODO: Insufficient memory allocation may occurs when writer is not used.
class Writer:
    """A tensorboard interface style writer.

    Arguments:
        mode: `tensorboard` or `wandb` can be chosen to determine how the information is presented.
    """

    def __init__(self, args):
        self.tensorboard_writer = None
        self.wandb_writer = None
        if hasattr(args, 'tensorboard_dir') and args.tensorboard_dir \
            and args.rank == (args.world_size - 1):
            try:
                from torch.utils.tensorboard import SummaryWriter
                print('> setting tensorboard ...')
                self.tensorboard_writer = SummaryWriter(
                    log_dir=args.tensorboard_dir,
                    max_queue=args.tensorboard_queue_size)
            except ModuleNotFoundError:
                print('WARNING: TensorBoard writing requested but is not '
                    'available (are you using PyTorch 1.1.0 or later?), '
                    'no TensorBoard logs will be written.', flush=True)
        wandb_project = os.environ.get('WANDB_PROJECT', None)
        if wandb_project is not None and args.rank == (args.world_size - 1):
            try:
                import wandb
                print('> setting wandb ...')
                wandb.login()
                self.wandb_writer = wandb.init()
                self.wandb_writer.config.update(args)
            except ModuleNotFoundError:
                print('WARNING: Wandb requested but is not '
                    'available, you should pip install wandb. No Wandb logs will be written.', flush=True)

    def add_scalar(self, tag, value, global_step, only_tensorboard=False):
        # assert self is not None, "no writer has been initialized"
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)
        if self.wandb_writer is not None and not only_tensorboard:
            self.wandb_writer.log(data={tag: value}, step=global_step)

    def add_text(self, tag, text, global_step):
        # assert self is not None, "no writer has been initialized"
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(tag=tag, text_string=text, global_step=global_step)
        if self.wandb_writer is not None:
            # no need to log this, config will show in wandb.config
            pass
    
    def finish(self):
        if self.wandb_writer is not None:
            print('> finish wandb ...')
            self.wandb_writer.finish()
            self.wandb_writer = None
        if self.tensorboard_writer is not None:
            print('> finish tensorboard ...')
            self.tensorboard_writer.close()
            self.tensorboard_writer = None
    
    def alert(self, *args, **kwargs):
        if self.wandb_writer is not None:
            self.wandb_writer.alert(*args, **kwargs)

    def __bool__(self):
        return self.tensorboard_writer is not None or self.wandb_writer is not None
