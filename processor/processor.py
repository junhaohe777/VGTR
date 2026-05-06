import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "js_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "mim_loss": AverageMeter(),
        "attr_loss": AverageMeter(),
        "memory_loss": AverageMeter(),
        "part_diversity_loss": AverageMeter(),
        "auxiliary_id_loss": AverageMeter(),
        
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        
        # Set memory epoch for warmup control
        if hasattr(model, 'module'):
            # Distributed training
            if hasattr(model.module, 'memory'):
                model.module.memory.set_epoch(epoch)
        else:
            # Single GPU training
            if hasattr(model, 'memory'):
                model.memory.set_epoch(epoch)
        
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            
            # Helper function to safely extract scalar value from loss
            def get_loss_value(loss_dict, key, default=0.0):
                value = loss_dict.get(key, default)
                if isinstance(value, torch.Tensor):
                    return value.item()
                return value
            
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(get_loss_value(ret, 'sdm_loss'), batch_size)
            meters['itc_loss'].update(get_loss_value(ret, 'itc_loss'), batch_size)
            meters['id_loss'].update(get_loss_value(ret, 'id_loss'), batch_size)
            meters['mlm_loss'].update(get_loss_value(ret, 'mlm_loss'), batch_size)
            meters['mim_loss'].update(get_loss_value(ret, 'mim_loss'), batch_size)
            meters['memory_loss'].update(get_loss_value(ret, 'memory_loss'), batch_size)
            meters['part_diversity_loss'].update(get_loss_value(ret, 'part_diversity_loss'), batch_size)
            meters['auxiliary_id_loss'].update(get_loss_value(ret, 'auxiliary_id_loss'), batch_size)
            
            meters['img_acc'].update(get_loss_value(ret, 'img_acc'), batch_size)
            meters['txt_acc'].update(get_loss_value(ret, 'txt_acc'), batch_size)
            meters['mlm_acc'].update(get_loss_value(ret, 'mlm_acc'), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
