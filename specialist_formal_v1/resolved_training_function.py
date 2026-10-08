def train_registered_job(job,dry=False):
    c=verify()
    import torch,transformers,peft,numpy as np
    from transformers import AutoTokenizer,TrainingArguments,Trainer,TrainerCallback,set_seed
    from peft import prepare_model_for_kbit_training,get_peft_model,LoraConfig
    set_seed(job["seed"]);torch.set_num_threads(4)
    folder=ROOT/"training"/job["name"];folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'COMPLETE.json').exists()
    t=AutoTokenizer.from_pretrained(c['models']['7B'],local_files_only=True);t.pad_token=t.eos_token
    rr=rows(ROOT/job["inputs"]);gg={g['request_id']:g for g in rows(ROOT/job["targets"])}
    items=[]
    for r in rr:
        pref=render(t,r,'zero',c)
        text=t.apply_chat_template(prompt(r,'zero',c)+[{'role':'assistant','content':canonical(payload(gg[r['request_id']]))}],tokenize=False,add_generation_prompt=False)
        ids=t(text,add_special_tokens=False)['input_ids'];pids=t(pref,add_special_tokens=False)['input_ids']
        assert ids[:len(pids)]==pids and len(ids)<=c['training_cap_tokens']
        items.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(pids)+ids[len(pids):]})
    if dry:items=sorted(items,key=lambda x:len(x['input_ids']),reverse=True)[:2]
    class DS(torch.utils.data.Dataset):
        def __len__(self):return len(items)
        def __getitem__(self,i):return items[i]
    def collate(batch):
        width=max(len(x['input_ids']) for x in batch);out={k:[] for k in ('input_ids','attention_mask','labels')}
        for x in batch:
            n=width-len(x['input_ids'])
            for k,pad in [('input_ids',t.pad_token_id),('attention_mask',0),('labels',-100)]:out[k].append(x[k]+[pad]*n)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in out.items()}
    model=get_peft_model(prepare_model_for_kbit_training(load(c['models']['7B'],True)),LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    h=hashlib.sha256()
    for k,p in model.named_parameters():
        if p.requires_grad:h.update(k.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    args=TrainingArguments(output_dir=str(folder/'state'),num_train_epochs=3,max_steps=2 if dry else -1,
        per_device_train_batch_size=1,gradient_accumulation_steps=1 if dry else 16,learning_rate=2e-4,lr_scheduler_type='linear',warmup_steps=0,
        logging_steps=1 if dry else 10,save_strategy='no',report_to='none',bf16=True,optim='paged_adamw_8bit',gradient_checkpointing=True,
        remove_unused_columns=False,seed=job["seed"],data_seed=job["seed"],dataloader_num_workers=0,disable_tqdm=True)
    meta={'protocol_sha256':sha(ROOT/'FROZEN_PROTOCOL.json'),'start':now(),'torch':torch.__version__,'transformers':transformers.__version__,
          'peft':peft.__version__,'gpu':torch.cuda.get_device_name(0),'training_args':args.to_dict(),'initial_trainable_sha256':h.hexdigest(),
          'tokenized_dataset_sha256':hashlib.sha256(canonical(items).encode()).hexdigest(),'n':len(items),'dryrun':dry,'no_truncation':True}
    dump(folder/'META.json',meta)
    class Timing(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):self.start=time.perf_counter()
        def on_step_end(self,args,state,control,**kw):
            if state.global_step%5==0 or dry or state.global_step==state.max_steps:
                elapsed=time.perf_counter()-self.start
                dump(folder/'PROGRESS.json',{'at':now(),'step':state.global_step,'total':state.max_steps,'elapsed_seconds':elapsed,
                     'estimated_remaining_seconds':elapsed/state.global_step*(state.max_steps-state.global_step),'peak_memory_MiB':torch.cuda.max_memory_allocated()/2**20})
    result=Trainer(model=model,args=args,train_dataset=DS(),data_collator=collate,callbacks=[Timing()]).train()
    assert math.isfinite(result.metrics['train_loss'])
    if not dry:
        adapter=folder/'adapter';adapter.mkdir(exist_ok=False);model.save_pretrained(adapter);t.save_pretrained(adapter)
    dump(folder/'COMPLETE.json',{'at':now(),'metrics':result.metrics,'passed':True,'adapter_sha256':None if dry else sha(adapter/'adapter_model.safetensors')})
    print('TRAIN_COMPLETE',dry,result.metrics,flush=True)
