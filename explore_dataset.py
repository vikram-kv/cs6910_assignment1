from keras.datasets import fashion_mnist, mnist
import wandb

datasets = [fashion_mnist, mnist]
names = ["fashion_mnist", "mnist"]
fashion_mnist_labels = {0 : 'T-shirt/top',  1	: 'Trouser', 2	: 'Pullover', 3	: 'Dress', 
4	: 'Coat', 5	: 'Sandal', 6	: 'Shirt', 7	: 'Sneaker', 8	: 'Bag', 9	: 'Ankle boot'}
mnist_labels = {0 : 'Zero',  1	: 'One', 2	: 'Two', 3	: 'Three', 
4	: 'Four', 5	: 'Five', 6	: 'Six', 7	: 'Seven', 8	: 'Eight', 9	: 'Nine'}
labels = [fashion_mnist_labels, mnist_labels]

for i in range(len(datasets)):
    (x_train, y_train), (x_test, y_test) = datasets[i].load_data()
    num_classes = len(labels[i])

    class_to_image = [[] for _ in range(num_classes)]
    for idx in range(x_train.shape[0]):
        class_to_image[y_train[idx]] += [idx]

    # plot 3 representative images in 3 separate runs
    for cnt in range(3):
        run = wandb.init(project='cs6910-assignment1', entity='cs19b021')
        run.name =f'images_{names[i]}_{cnt+1}'
        columns=["Class name 1", "Image 1", "Class name 2", "Image 2"]
        test_table = wandb.Table(columns=columns)
        for idx in range(5):
            test_table.add_data(f"[{idx}] or [{labels[i][idx]}]", wandb.Image(x_train[class_to_image[idx][cnt]]),
                                f"[{idx+5}] or [{labels[i][idx+5]}]", wandb.Image(x_train[class_to_image[idx+5][cnt]]))
        wandb.log({f"table" : test_table})
        run.finish()    
