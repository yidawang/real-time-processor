import pyinotify
import os
import sys
import logging
import time
import argparse
from process_single_tr import SingleTRProcessor


format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='xxx.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

class EventHandler(pyinotify.ProcessEvent):
    def __init__(self, stp):
        self.stp = stp
        self.tr_count = 0

    def process_IN_CLOSE_WRITE(self, event):
        #start = datetime.datetime.utcnow()
        filepath = event.pathname
        filename = filepath[filepath.rindex('/')+1:]
        # read data and process it
        time1 = time.time()
        self.stp.process_single_tr_nifti(filepath, self.tr_count)
        time2 = time.time()
        logger.info(
                '%s is processed in %.2f ms' %
                (filename, (time2 - time1) * 1000)
                )
        self.tr_count += 1


class NiftiFileWatcher():
    def __init__(self, options):
        self.path = options.get('directory')
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            logger.info(self.path + ' dost not exist ==> creating...')
        elif not os.path.isdir(self.path):
            logger.error('path argument must be a directory')
            sys.exit(1)
        logger.setLevel(getattr(logging, (options.get('loglevel') or 'info').upper()))
        epoch_file = options.get('epoch')
        mask_file = options.get('mask')
        window = options.get('window') or 8
        total = options.get('total') or 1500
        stp = SingleTRProcessor(epoch_file, mask_file, total, window)
        self.handler = EventHandler(stp)

    def watch_dir(self):
        mask = pyinotify.IN_CLOSE_WRITE
        wm = pyinotify.WatchManager()
        notifier = pyinotify.Notifier(wm, self.handler)
        wm.add_watch(self.path, mask)
        notifier.loop()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('directory', help='path to the monitored directory')
    arg_parser.add_argument('epoch', help='path to the numpy file of epoch info in 1D array')
    arg_parser.add_argument('mask', help='path to the mask file which specifies the top voxels')
    arg_parser.add_argument('-l', '--loglevel', help='log level [INFO]')
    arg_parser.add_argument('-w', '--window', type=int, help='correlation window size [8]')
    arg_parser.add_argument('-t', '--total', type=int, help='total number of TRs generated in real-time [1500]')

    args = arg_parser.parse_args()

    input_dict = vars(args)

    nifti_file_watcher = NiftiFileWatcher(input_dict)
    nifti_file_watcher.watch_dir()

