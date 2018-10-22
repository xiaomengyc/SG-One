# import caffe
import numpy as np
import random
from multiprocessing import Process, Queue, Pool, Lock

import sys
import traceback
import util
from util import cprint, bcolors
from operator import itemgetter
from functools import partial
from skimage.transform import resize
import copy

import os.path as osp
import os
import multiprocessing
import multiprocessing.pool
from skimage.io import imsave

import  ss_settings as settings

class DBInterface():
    def __init__(self, params):
        self.lock = Lock()
        self.params = params
        self.load_items()

        # initialize the random generator
        self.init_randget(params['read_mode'])
        self.cycle = 0
        
    def init_randget(self, read_mode):
        self.rand_gen = random.Random()
        if read_mode == 'shuffle':
            self.rand_gen.seed()
        elif read_mode == 'deterministic':
            self.rand_gen.seed(1385) #>>>Do not change<<< Fixed seed for deterministic mode. 
    
    def update_seq_index(self):
        self.seq_index += 1
        if self.seq_index >= len(self.db_items):
            self.db_items = copy.copy(self.orig_db_items)
            self.rand_gen.shuffle(self.db_items)
            self.seq_index = 0
    
    def next_pair(self):
        with self.lock:
            end_of_cycle = self.params.has_key('db_cycle') and self.cycle >= self.params['db_cycle']
            if end_of_cycle:
                assert(self.params['db_cycle'] > 0)
                self.cycle = 0
                self.seq_index = len(self.db_items)
                self.init_randget(self.params['read_mode'])
                
            self.cycle += 1
            base_trans = None if self.params['image_base_trans'] is None else self.params['image_base_trans'].sample()
            self.update_seq_index()
            if self.params['output_type'] == 'single_image':
                db_item = self.db_items[self.seq_index]
                assert(isinstance(db_item, util.DBImageItem))
                player = util.ImagePlayer(db_item, base_trans, None, None, length = 1)
                return player, [0], None
            elif self.params['output_type'] == 'image_pair':
                imgset, second_index = self.db_items[self.seq_index]
                player = util.VideoPlayer(imgset, base_trans, self.params['image_frame_trans'])
                set_indices = range(second_index) + range(second_index+1, player.length)
                assert(len(set_indices) >= self.params['k_shot'])
                self.rand_gen.shuffle(set_indices)
                first_index = set_indices[:self.params['k_shot']]
                return player, first_index, second_index
            else:
                raise Exception('Only single_image and image_pair mode are supported')
    
    def _remove_small_objects(self, items):
        filtered_item = []
        for item in items:
            mask = item.read_mask()
            if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
                filtered_item.append(item)
        return filtered_item
    
    def load_items(self):
        self.db_items = []
        if self.params.has_key('image_sets'):
            for image_set in self.params['image_sets']:
                print image_set
                if image_set.startswith('pascal') or image_set.startswith('sbd'):
                    if image_set.startswith('pascal'):
                        pascal_db = util.PASCAL(self.params['pascal_path'], image_set[7:])   
                    elif image_set.startswith('sbd'):
                        pascal_db = util.PASCAL(self.params['sbd_path'], image_set[4:])   
                    #reads single image and all semantic classes are presented in the label

                    if self.params['output_type'] == 'single_image':
                        items = pascal_db.getItems(self.params['pascal_cats'], self.params['areaRng'],
                                                   read_mode = util.PASCAL_READ_MODES.SEMANTIC_ALL)
                    #reads pair of images from one semantic class and and with binary labels
                    elif self.params['output_type'] == 'image_pair':
                        items = pascal_db.getItems(self.params['pascal_cats'], self.params['areaRng'],
                                                   read_mode = util.PASCAL_READ_MODES.SEMANTIC)
                        items = self._remove_small_objects(items)
                    else:
                        raise Exception('Only single_image and image_pair mode are supported')
                    self.db_items.extend(items)
                else:
                    raise Exception
            cprint('Total of ' + str(len(self.db_items)) + ' db items loaded!', bcolors.OKBLUE)
            
            #reads pair of images from one semantic class and and with binary labels
            if self.params['output_type'] == 'image_pair':
                items = self.db_items
                
                #In image_pair mode pair of images are sampled from the same semantic class
                clusters = util.PASCAL.cluster_items(self.db_items)
                
                #for set_id in clusters.keys():
                #    print clusters[set_id].length
                
                #db_items will be a list of tuples (set,j) in which set is the set that img_item belongs to and j is the index of img_item in that set
                self.db_items = []
                for item in items:
                    set_id = item.obj_ids[0]
                    imgset = clusters[set_id]
                    assert(imgset.length > self.params['k_shot']), 'class ' + imgset.name + ' has only ' + imgset.length + ' examples.'
                    in_set_index = imgset.image_items.index(item)
                    self.db_items.append((imgset, in_set_index))
                cprint('Total of ' + str(len(clusters)) + ' classes!', bcolors.OKBLUE)
        
        
        self.orig_db_items = copy.copy(self.db_items)

        assert(len(self.db_items) > 0), 'Did not load anything from the dataset'
        #assert(not self.params.has_key('db_cycle') or len(self.db_items) >= self.params['db_cycle']), 'DB Cycle should can not be more than items in the database = ' + str(len(self.db_items))
        #it forces the update_seq_index function to shuffle db_items and set seq_index = 0
        self.seq_index = len(self.db_items)
            

class PairLoaderProcess(Process):
    def __init__(self, name, queue, db_interface, params):
        Process.__init__(self, name=name)
        self.queue = queue
        self.db_interface = db_interface
        self.first_shape = params['first_shape']
        self.second_shape = params['second_shape']
        if params.has_key('shape_divisible'):
            self.shape_divisible = params['shape_divisible']
        else:
            self.shape_divisible = 1
        
        self.bgr = params['bgr']
        self.scale_256 = params['scale_256']
        self.first_label_mean = params['first_label_mean']
        self.first_label_scale = 1.0 if not params.has_key('first_label_scale') else params['first_label_scale']
        self.mean = np.array(params['mean']).reshape(1,1,3)
        self.first_label_params = params['first_label_params']
        self.second_label_params = params['second_label_params']
        self.deploy_mode = params['deploy_mode'] if params.has_key('deploy_mode') else False
        self.has_cont = params['has_cont'] if params.has_key('has_cont') else False
        if self.bgr:
            #Always store mean in RGB format
            self.mean = self.mean[:,:, ::-1]
            
    def run(self):
        try:
            while True:
                item = None
                while item is None:
                    item = self.load_next_frame()
                self.queue.put(item)
        except:
            cprint('An Error Happended in run()',bcolors.FAIL)
            cprint(str("".join(traceback.format_exception(*sys.exc_info()))), bcolors.FAIL)
            self.queue.put(None)
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    
    def load_next_frame(self, try_mode=True):
        next_pair = self.db_interface.next_pair()
        item = self.load_frame(*next_pair)
        # try to load the image more times in case it is None
        if item is None and not try_mode:
            item = self.try_some_more(100)
        return item

    # Tries to look for a valid image for a limited number of tries and then  
    # returns None if it doesn't find it
    def try_some_more(self, max_tries):
        i=0
        item =None
        while(item is None and i<max_tries):
            item = self.load_next_frame(True)
            i+=1
            print 'Skipping image because of tiny object'
        return item

    def __prepross(self, frame_dict, shape = None):
        if frame_dict['mask'] is None: 
            return None
        
        # image = frame_dict['image'] - self.mean
        image = frame_dict['image']
        label = frame_dict['mask']

        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)
        image = (image-mean_vals)/std_vals

        if shape is None:
            shape = np.array(image.shape[:-1], dtype=int)
        if self.shape_divisible != 1:
            shape = np.array(self.shape_divisible * np.ceil(shape / self.shape_divisible), dtype=np.int)     
        
        if tuple(shape) != image.shape[:-1]:
            image = resize(image, shape)
            label = resize(label, shape, order = 0, preserve_range=True)
            
        # if self.bgr:
        #     image = image[:,:, ::-1]
            
        if self.scale_256:
            image *= 255
            
        return image, label, shape
    
    def __is_integer(self, mask):
      label_set = np.array(np.unique(mask), dtype=float)
      for label in label_set:
          if not label.is_integer():
              return False
      return True
    
    def __get_deploy_info(self, player, index):
        if index is None:
            return None, None, None
        if isinstance(player, util.ImagePlayer):
            img_item = player.image_item
            return img_item.obj_ids, img_item.read_mask(True), img_item.read_img()
        elif isinstance(player, util.VideoPlayer):
            img_item = player.video_item.image_items[index]
            return img_item.obj_ids, img_item.read_mask(True), img_item.read_img()
        else:
            raise Exception
    
    def load_frame(self, player, first_index, second_index):
        cprint('Loading pair = ' + player.name + ', ' + str(first_index) + ', ' + str(second_index), bcolors.WARNING)
        if second_index in first_index:
            return None
        
        
        images1 = []
        labels1 = []
        image_path = []
        shape1 = self.first_shape
        for ind in first_index:
            frame1_dict = player.get_frame(ind)
            image1, label1, shape1 = self.__prepross(frame1_dict, shape1)
            images1.append(image1.transpose((2,0,1)))
            labels1.append(label1)
            image_path.append(frame1_dict['image_path'])
        item = dict(first_img=images1, image1_path=image_path)

        if second_index is not None:
            frame2_dict = player.get_frame(second_index)
            image2, label2, shape = self.__prepross(frame2_dict, self.second_shape)
            item['second_img'] = [image2.transpose((2,0,1))]
            item['image2_path'] = [frame2_dict['image_path'],]
        
        
        if self.deploy_mode:
            first_semantic_labels=[]
            first_mask_orig=[]
            first_img_orig=[]
            for ind in first_index:
                a,b,c = self.__get_deploy_info(player, ind)
                first_semantic_labels.append(a)
                first_mask_orig.append(b)
                first_img_orig.append(c)

            deploy_info = dict(seq_name=player.name, 
                               first_index=first_index, 
                               first_img_orig=first_img_orig,
                               first_mask_orig=first_mask_orig,
                               first_semantic_labels=first_semantic_labels)
            
            if second_index is not None:
                second_semantic_labels, second_mask_orig, second_img_orig = self.__get_deploy_info(player, second_index)
                deploy_info.update(second_index=second_index,
                                   second_img_orig=second_img_orig,
                                   second_mask_orig=second_mask_orig,
                                   second_semantic_labels=second_semantic_labels)
            
            item['deploy_info'] =  deploy_info
        
        #create first_labels
        for i in range(len(self.first_label_params)):
            name, down_scale, offset = self.first_label_params[i]
            item[name] = []
            for label1 in labels1:
                nlabel1 = util.change_coordinates(label1, down_scale, offset)
                nlabel1 = (nlabel1 - self.first_label_mean) * self.first_label_scale
                assert(self.__is_integer(nlabel1))
                item[name].append(nlabel1.reshape((1,) + nlabel1.shape))

        if second_index is not None:
            #create second_labels
            for i in range(len(self.second_label_params)):
                name, down_scale, offset = self.second_label_params[i]
                nlabel2 = util.change_coordinates(label2, down_scale, offset)
                assert(self.__is_integer(nlabel2))
                item[name] = [nlabel2.reshape((1,) + nlabel2.shape)]
        if self.has_cont:
            item['cont'] = [0] + [1] * (len(first_index) - 1)
            
        return item
            
# class SSDatalayer(caffe.Layer):
class SSDatalayer():
    def __init__(self, test_group, k_shot=1):
        # for process in self.processes:
        #     process.terminate()

        self.param_str = {
            # 'profile':'fold%d_1shot_test'%(test_group),
            'profile':'fold%d_%dshot_test'%(test_group,k_shot),
            'second_label_params':[('first_label', 1.0, 0.0)],
            'first_label_params':[('second_label', 1.0, 0.0)],
            'batch_size': 1,
            # 'k_shot':1,
            'k_shot':k_shot,
            'has_cont':False,
            'deploy_mode':True,
            'pascal_path':'/home/xiaolin/xlzhang/data/VOC2012'
        }

        self.setup()

    def setup(self):
        params = self.param_str
         
        if params.has_key('profile'):
            # print "import settings"
            settings = __import__('ss_settings')
            profile = getattr(settings, params['profile'])
            profile.update(params)
            params = profile


        self.all_top_names = ['first_img', 'second_img']

        if params.has_key('top_names'):
            self.top_names = copy.copy(params['top_names'])
        else:
            self.top_names = copy.copy(self.all_top_names)

        assert(set(self.top_names) <= set(self.all_top_names)), str(self.top_names) + ' is not subset of ' + str(self.all_top_names)
        
        if params.has_key('has_cont') and params['has_cont']:
            self.top_names.append('cont')
            
        for i in range(len(params['first_label_params'])):
            name, down_scale, offset = params['first_label_params'][i]
            self.top_names.append(name)

        for i in range(len(params['second_label_params'])):
            name, down_scale, offset = params['second_label_params'][i]
            self.top_names.append(name)
        
        self.batch_size = params['batch_size']

        self.queue = Queue(self.batch_size * params['worker_num'])
        self.db_interface = DBInterface(params)
        self.processes = []
        for i in range(params['worker_num']):
            process = PairLoaderProcess('SSDatalayer Process', self.queue, self.db_interface, params)
            process.daemon = True
            process.start()
            self.processes.append(process)

    ###### Queue operations
    def init_queue(self):
        if not hasattr(self, 'cur_item') or self.cur_item is None:
            self.cur_item = self.queue.get()
            
    def dequeue(self):
        self.init_queue()
        item = self.cur_item
        self.cur_item = None
        self.init_queue()
        return item    


