# Sentiment Analysis With Deep Learning Using BERT
This project is credited to https://www.coursera.org/projects/sentiment-analysis-bert

The learning objectives of this project are as follows:
- What BERT is and what it can do
- Clean and preprocess text dataset
- Split dataset into training and validation sets using stratified approach
- Tokenize (encode) dataset using BERT toknizer
- Design BERT finetuning architecture
- Evaluate performance using F1 scores and accuracy
- Finetune BERT using training loop

## Main Training Loop
```
for epoch in tqdm(range(1, epochs+1)):
  model.train()
  loss_train_total = 0
  progress_bar = tqdm(dataloader_train,
                      desc = 'Epoch {:1d}'.format(epoch),
                      leave= False,
                      disable = False)
  for batch in progress_bar:
    model.zero_grad()

    batch = tuple(b.to(device) for b in batch)
    
    inputs = {
        'input_ids' : batch[0],
        'attention_mask': batch[1],
        'labels': batch[2]
    }

    outputs = model(**inputs)
    loss = outputs[0]
    loss_train_total += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
  
  torch.save(model.state_dict(),f'/content/Models/BERT_ft_epoch{epoch}.model')
  tqdm.write('\nEpoch {epoch}')
  loss_train_avg = loss_train_total/len(dataloader_train)
  tqdm.write(f'Training loss: {loss_train_avg}')
  val_loss, predictions, true_vals = evaluate(dataloader_val)
  val_f1 = f1_score_func(predictions, true_vals)
  tqdm.write(f'Validation loss:{val_loss}')
  tqdm.write(f'F1 Score (weighted): {val_f1}')
  
```
## Results
Class: happy
Accuracy: 161/ 171

Class: not-relevant
Accuracy: 22/ 32

Class: angry
Accuracy: 8/ 9

Class: disgust
Accuracy: 0/ 1

Class: sad
Accuracy: 0/ 5

Class: surprise
Accuracy: 0/ 5
