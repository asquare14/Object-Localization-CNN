from torch.utils.data import Dataset

class dataset(Dataset):

    def __init__(self, df , root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        img_name = os.path.join(self.root_dir, self.df.iloc[index, 0])
        image = cv2.imread(img_name)
        box = np.array([self.df.iloc[index, 1], self.df.iloc[index, 3], self.df.iloc[index, 2], self.df.iloc[index, 4]])
        image = image.reshape(3,640,480)
        return {'image': torch.from_numpy(image).type(torch.DoubleTensor).double().cuda(),
                'box': torch.from_numpy(box)}
