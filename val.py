r"""
For validating model.
Consist of some Valers.
"""

from utils import \
    LOGGER, WRITER, \
    compute_fps, \
    ValDetectMixin

__all__ = ['ValDetect']


class ValDetect(
    ValDetectMixin  # for validating
):
    def __init__(self, args=None, last=True,
                 model=None, half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None, writer=None):
        super(ValDetect, self).__init__()
        self.last = last
        self.training = model is not None
        self.time = 0.0
        self.seen = 0
        self.writer = writer

        if self.training:
            # val during training
            self.device = next(model.parameters()).device
            self.half = half
            self.loss_fn = loss_fn
            self.dataloader = dataloader
            if self.half and self.device.type == 'cpu':
                LOGGER.warning(f'The device is {self.device}, half precision only supported on CUDA')
                self.half = False
            self.model = model.half() if self.half else model.float()
            self.cls_names = cls_names
            self.epoch = epoch
        else:
            # val alone
            pass

    def val(self):
        self.model.eval()
        # TODO maybe save something or plot images below
        loss_all, loss_name, stats = self.val_once()
        # TODO move the compute_metrics and compute_fps to Mixin in the future for upgrading
        metrics = self.compute_metrics(stats)
        fps_time = compute_fps(self.seen, self.time)
        # TODO confusion matrix needed
        # TODO get the stats of the target number per class which detected correspond to label correctly
        self._log_results(loss_all, loss_name, metrics, fps_time)
        if not self.last:
            self.model.float()
        return (loss_all, loss_name), metrics, fps_time

    def _log_results(self, loss_all, loss_name, metrics, fps_time):
        """
        metrics = (ap_all, f1_all, p_all, r_all, cls_name_number)

        ap_all = (ap50_95, ap50, ap75, ap)
        f1_all = (mf1, f1)
        p_all = (mp, p)
        r_all = (mr, r)
        cls_name_number = (cls, cls_number)
        """
        separate = '-' * 60
        t_fmt = '<15'  # title format
        fmt = t_fmt + '.3f'
        space = ' ' * 50

        # speed
        if self.last:
            LOGGER.info(f'{space}Speed {fps_time[1]:.2f} ms per image, FPs: {fps_time[0]:.1f}, accuracy')
        else:
            LOGGER.info(f'{space}Speed {fps_time[1]:.2f} ms per image, FPs: {fps_time[0]:.1f}, no accuracy')

        (ap50_95, ap50, ap75, ap), (mf1, f1), (mp, p), (mr, r), (cls, cls_number) = metrics
        if ap is not None:
            name_writer = ('P_50', 'R_50', 'F1_50', 'AP50', 'AP75', 'AP5095')
            value_writer = (mp[0], mr[0], mf1[0], ap50, ap75, ap50_95)
            WRITER.add_epoch_curve(self.writer, 'val_metrics', value_writer, name_writer, self.epoch)

            LOGGER.debug(f'{separate}')
            LOGGER.debug(f'Validating epoch{self.epoch}: {loss_name} is {loss_all}')
            LOGGER.debug(f'P_50: {mp[0]}, R_50: {mr[0]}, F1_50: {mf1[0]}, '
                         f'AP50: {ap50}, AP75: {ap75}, AP/AP5095: {ap50_95}')
            LOGGER.debug(f'P_5095: {mp}, R_5095: {mr}, F1_5095: {mf1}')

            rest = (cls, ap, f1, p, r)
            LOGGER.debug(f'cls_name, cls_number, (IoU=0.50:0.95) AP, F1, P, R')
            for c, ap_c, f1_c, p_c, r_c in zip(*rest):
                name_c = self.cls_names[c]
                number_c = cls_number[c]
                LOGGER.debug(f'{name_c}, {number_c}, {ap_c}, {f1_c}, {p_c}, {r_c}')
            LOGGER.debug(f'{separate}')

            if self.last:
                LOGGER.info(f"{'class_name':{t_fmt}}"
                            f"{'class_number':{t_fmt}}"
                            f"{'R':{t_fmt}}"
                            f"{'P':{t_fmt}}"
                            f"{'F1':{t_fmt}}"
                            f"{'AP50':{t_fmt}}"
                            f"{'AP75':{t_fmt}}"
                            f"{'AP/AP5095':{t_fmt}}")
                for c, ap_c, f1_c, p_c, r_c in zip(*rest):
                    name_c = self.cls_names[c]
                    number_c = cls_number[c]
                    LOGGER.info(f'{name_c:{t_fmt}}'
                                f'{number_c:{t_fmt}}'
                                f'{r_c[0]:{fmt}}'
                                f'{p_c[0]:{fmt}}'
                                f'{f1_c[0]:{fmt}}'
                                f'{ap_c[0]:{fmt}}'
                                f'{ap_c[5]:{fmt}}'
                                f'{sum(ap_c) / len(ap_c):{fmt}}')
        else:
            name_writer = ('P_50', 'R_50', 'F1_50', 'AP50', 'AP75', 'AP5095')
            value_writer = (0,) * 6
            WRITER.add_epoch_curve(self.writer, 'val_metrics', value_writer, name_writer, self.epoch)
            LOGGER.debug(f'{separate}')
            LOGGER.debug(f'Validating epoch{self.epoch}: {loss_name} is {loss_all}')
            LOGGER.debug(f'others is None, cls number is {cls_number}')
            LOGGER.debug(f'{separate}')


if __name__ == '__main__':
    pass
