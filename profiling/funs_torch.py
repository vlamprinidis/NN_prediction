import torch
import csv

def train(model, train_loader, epochs, criterion, optimizer):
    total_step = len(train_loader)
    print(total_step)
    
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            labels = labels

            # Forward pass
            outputs = model(images)            
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                       .format(epoch+1, epochs, i+1, total_step, loss))

def give_evt_lst(evt):
    return [evt.key, str(evt.cpu_time), 
            str(evt.cpu_time_total), str(evt.self_cpu_time_total), str(evt.count)]

def save_to_csv(function_events, name):
    headers = [
        'Name',
        'CPU time avg (us)',
        'CPU total (us)',
        'Self CPU time total (us)',
        'Number of Calls'
    ]
    with open(name, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        #way to write to csv file
        writer.writerow(headers)
        for evt in function_events:
            writer.writerow( give_evt_lst(evt) )

# def get_torch_ops():
