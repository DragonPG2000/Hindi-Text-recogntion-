img_dir='/content/drive/My Drive/Projects/Handwriting recognizer/Images/'
model_dir='/content/drive/My Drive/Projects/Handwriting recognizer/Models/'
img_height=80*3
img_width=180*3 
input_shape=(img_width,img_height,3)
batch_size=16
max_len=5
lr=1e-3
base_lr=1e-4
max_lr=3e-4 #To the power of 4 for best 
epochs=25
hidden_size=1024  
model_name='bn_vgg19b' 

vocab=['', 'ै', 'ॉ', 'अ', 'ञ', '७', 'ौ', '्', 'द', '़', 'क', 'ँ', 'ऐ', 'ष', 'फ', 'ई', 'घ', 'ग', '९', 'औ', 'े', '१', 'ध', 'इ', 'n', '\u200c', 'छ', 'ा', 'य', 'l', 'च', 'y', 'स', 'ब', '"', 'त', 'ओ', 's', '-', 'F', '|', 'व', 'झ', 'ॅ', '६', 'म', 'र', 'ी', 'थ', 'ृ', 't', 'r', '(', 'उ', 'ऋ', 'भ', '\u200d', 'न', 'ळ', '०', 'u', 'i', 'ए', 'a', 'G', ')', 'e', 'ठ', 'ण', '५', '३', '२', 'ल', 'ऊ', 'ॐ', '४', 'ू', 'R', 'श', "'", 'm', 'ु', 'ः', 'ख', 'ज', 'ऑ', 'आ', ':', 'ढ', 'ट', 'प', 'ड', 'ं', 'ह', 'ि', '८', 'ो']
n_classes=len(vocab)+1

