from nsfw_detector import NSFWDetector

# detector = NSFWDetector('./models/nsfw.299x299.h5')
detector_mobilenet = NSFWDetector('./models/nsfw_mobilenet2.224x224.h5')

# Predict single image
# preds = detector.predict('./images/2.jpeg')
# print(preds)
# {'./images/1.jpeg': {'hentai': 0.00039620648, 'drawings': 0.0015517682, 'porn': 0.0145489825, 'neutral': 0.47937542, 'sexy': 0.5041276}}
# {'./images/2.jpeg': {'sexy': 0.002209452, 'porn': 0.02362343, 'neutral': 0.08266616, 'hentai': 0.104420066, 'drawings': 0.7870809}}
# {'./images/3.jpeg': {'drawings': 0.0010201228, 'hentai': 0.00948968, 'neutral': 0.112305455, 'sexy': 0.20929794, 'porn': 0.66788685}}
# {'./images/4.jpg': {'drawings': 0.0039954395, 'hentai': 0.00865483, 'sexy': 0.014153672, 'neutral': 0.048371613, 'porn': 0.9248244}}
# {'./images/5.jpg': {'sexy': 1.2161224e-05, 'neutral': 3.0270021e-05, 'porn': 0.009675577, 'drawings': 0.024789723, 'hentai': 0.9654921}}
# {'./images/6.png': {'drawings': 3.810143e-05, 'neutral': 0.0012048744, 'hentai': 0.0028310823, 'sexy': 0.0052776327, 'porn': 0.9906482}}

# Predict multiple images at once using Keras batch prediction
# preds = detector.predict(['./images/1.jpeg', './images/2.jpeg', './images/3.jpeg'], batch_size = 32)
# print(preds)

# Predict single image using mobilenet
preds = detector_mobilenet.predict('./images/6.png', image_size=(224,224))
print(preds)
# {'./images/1.jpeg': {'hentai': 0.001744987, 'drawings': 0.0048740534, 'porn': 0.038519483, 'sexy': 0.2532822, 'neutral': 0.7015793}}
# {'./images/2.jpeg': {'sexy': 0.0047700647, 'porn': 0.00802055, 'hentai': 0.031785157, 'drawings': 0.43713486, 'neutral': 0.5182893}}
# {'./images/3.jpeg': {'drawings': 0.0048237047, 'hentai': 0.036083117, 'neutral': 0.040656526, 'sexy': 0.38874716, 'porn': 0.52968943}}
# {'./images/4.jpg': {'drawings': 0.028689459, 'hentai': 0.034322977, 'sexy': 0.043128658, 'porn': 0.35343596, 'neutral': 0.5404229}}
# {'./images/5.jpg': {'neutral': 0.00012503797, 'sexy': 0.00017660983, 'drawings': 0.0043948903, 'porn': 0.029123513, 'hentai': 0.96618}}
# {'./images/6.png': {'drawings': 3.2457173e-09, 'neutral': 4.2654e-06, 'hentai': 0.00070470053, 'sexy': 0.025317859, 'porn': 0.97397316}}