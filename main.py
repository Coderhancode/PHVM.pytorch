import os
import sys
import numpy as np
import json
import argparse
import Config
import pickle
import torch
from torch.utils.data import DataLoader
import utils

import Dataset
import time

from Models import PHVM

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def import_lib():
	global Dataset, utils, device_lib, PHVM, Dataset

def dump(texts, filename):
	file = open(filename, "w")
	for inst in texts:
		lst = []
		for sent in inst:
			sent = " ".join(sent)
			lst.append({'desc': sent})
		file.write(json.dumps(lst, ensure_ascii=False) + "\n")
	file.close()

def agg_group(stop, text, start_token, end_token):
    translation = []
    for gcnt, sent in zip(stop, text):
        sent = sent[:gcnt, :]
        desc = []
        for segId, seg in enumerate(sent):  # 将每个句子的所有group的词ID整合到一起
            for wid in seg:
                if wid == end_token:
                    break
                elif wid == start_token:
                    continue
                else:
                    desc.append(wid)
        translation.append(desc)    # 句子所有词ID的集合
    max_len = 0
    for sent in translation:
        max_len = max(max_len, len(sent))
    for i, sent in enumerate(translation):
        translation[i] = [sent + [end_token] * (max_len - len(sent))]
    return translation

def infer(model, dataset):
    brand_set = pickle.load(open(config.brand_set_file, "rb"))
    vocab = dataset.vocab
    test_dataloader = DataLoader(dataset, batch_size=config.test_batch_size)
    test_iter = iter(test_dataloader)
    res = []
    while True:
        try:
            key_input, val_input, input_lens, target_input, target_output, output_lens, groups, glens, group_cnt, target_type, target_type_lens, text, slens, cate_input  = next(test_iter)
            
            sent_idx, stop = model(key_input.cuda(), val_input.cuda(), input_lens.cuda(), target_input.cuda(), target_output.cuda(), output_lens.cuda(), groups.cuda(), glens.cuda(), group_cnt.cuda(), text.cuda(), cate_input.cuda())
            print(sent_idx, stop)
            output = agg_group(stop.cpu().numpy().astype(int), sent_idx.cpu().numpy().astype(int), 0, 1)
            _output = []
            for inst_id, inst in enumerate(output):
                sents = []
                dup = set()
                for beam in inst:
                    sent = []
                    for wid in beam:
                        if wid == dataset.vocab.end_token:
                            break
                        elif wid == dataset.vocab.start_token:
                            continue
                        sent.append(vocab.id2word[wid] if vocab.id2word[wid] not in brand_set else "BRAND")
                    if str(sent) not in dup:
                        dup.add(str(sent))
                        sents.append(sent)
                _output.append(sents)
            print(_output)
            res.extend(_output)
            break
        except StopIteration:
            pass
    return res

def evaluate(model, dataset, data):
	batch = dataset.get_batch(data)
	tot_loss = 0
	tot_cnt = 0
	while True:
		try:
			batchInput = dataset.next_batch(batch)
			global_step, loss = model.eval(batchInput)
			slens = batchInput.slens
			tot_cnt += len(slens)
			tot_loss += loss * len(slens)
		except tf.errors.OutOfRangeError:
			break
	return tot_loss / tot_cnt

def _train(model_name, model, optimizer, lr_scheduler, train_loader, init):
    train_iter = iter(train_loader)
    model.train()
    
    for epoch in range(config.epoch):
        i = 0
        start_time = time.time()
        while i < len(train_loader):
            try:
                key_input, val_input, input_lens, target_input, target_output, output_lens, groups, glens, group_cnt, target_type, target_type_lens, text, slens, cate_input = next(train_iter)
                #print(cate_input)
                loss = model(key_input.cuda(), val_input.cuda(), input_lens.cuda(), target_input.cuda(), \
                            target_output.cuda(), output_lens.cuda(), groups.cuda(), glens.cuda(), \
                            group_cnt.cuda(), text.cuda(), cate_input.cuda())
                if i != 0 and i % 100 == 0:
                    end_time = time.time()
                    print("loss:", loss.item(), "epoch:", epoch, \
                    "iter_num:", i, "/", len(train_loader), \
                    "lr:", optimizer.state_dict()['param_groups'][0]['lr'], \
                    "cost_time_100_iters:", end_time-start_time)
                    #sys.stdout.flush()
                    start_time = time.time()
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                torch.cuda.empty_cache()
                i += 1
            except StopIteration:
                train_iter = iter(train_loader)
        
        lr_scheduler.step()
        torch.save(model.state_dict(), config.checkpoint_dir+'/'+model_name+'/'+str(epoch)+'.pth')


def train(model_name, restore=True):
    import_lib()
    dataset = Dataset.EPWDataset(config.train_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(dataset.vocab.id2word)
    train_dataloader = DataLoader(dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True,
        collate_fn=Dataset.alignCollate())
    logger = utils.get_logger(model_name)

    model = PHVM.PHVM(True, config.train_batch_size, device, len(dataset.vocab.id2featCate), len(dataset.vocab.id2featVal), 
                      len(dataset.vocab.id2word), len(dataset.vocab.id2category), len(dataset.vocab.id2type),
                      key_wordvec=None, val_wordvec=None, tgt_wordvec=dataset.vocab.id2vec)
    if torch.cuda.is_available():
        model.cuda()    
    '''
    for name, param in model.named_parameters():
        if name.startswith("weight"):
            nn.init.xavier_normal_(param)
        else:
            nn.init.zeros_(param)
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    init = {'epoch': 0, 'worse_step': 0}
    '''
    if restore:
        init['epoch'], init['worse_step'], model = model_utils.restore_model(model,
                                            config.checkpoint_dir + "/" + model_name + config.tmp_model_dir,
                                            config.checkpoint_dir + "/" + model_name + config.best_model_dir)
    '''
    config.check_ckpt(model_name)
    #summary = tf.summary.FileWriter(config.summary_dir, model.graph)
    _train(model_name, model, optimizer, lr_scheduler, train_dataloader, init)
    logger.info("finish training {}".format(model_name))

def get_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda x : x.lower() == 'true')
	parser.add_argument("--cuda_visible_devices", type=str, default='0,1,2,3')
	parser.add_argument("--train", type="bool", default=True)
	parser.add_argument("--restore", type="bool", default=False)
	parser.add_argument("--model_name", type=str, default="PHVM")
	parser.add_argument("--checkpoint", type=str, default="./result/checkpoint/PHVM/20.pth")
	args = parser.parse_args(sys.argv[1:])
	return args

def main():
	args = get_args()
	global config, logger
	config = Config.config

	if args.train:
		import_lib()
		train(args.model_name)
	else:
		import_lib()
		dataset = Dataset.EPWDataset(config.test_file)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = PHVM.PHVM(False, config.test_batch_size, device, len(dataset.vocab.id2featCate), len(dataset.vocab.id2featVal), 
						  len(dataset.vocab.id2word), len(dataset.vocab.id2category), len(dataset.vocab.id2type),
						  key_wordvec=None, val_wordvec=None, tgt_wordvec=dataset.vocab.id2vec)

		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint)
		if torch.cuda.is_available():
			model.cuda()
		
		texts = infer(model, dataset)
        for 
		dump(texts, config.result_dir + "/{}.json".format(args.model_name))
		utils.print_out("finish file test")

if __name__ == "__main__":
	main()
