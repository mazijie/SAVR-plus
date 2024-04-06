import os
import random
import cairosvg


def make_dataset(svg_dir, to_dir, train2eval_ratio=5, to_svg=False, output_size=(224, 224)):
    if os.path.exists(to_dir) or not os.path.exists(svg_dir):
        print('ERROR!')
        return
    os.mkdir(to_dir)
    os.mkdir(os.path.join(to_dir, 'train'))
    os.mkdir(os.path.join(to_dir, 'eval'))
    for chart_type in os.listdir(svg_dir):
        from_path = os.path.join(svg_dir, chart_type)
        to_path_train = os.path.join(to_dir, 'train', chart_type)
        to_path_eval = os.path.join(to_dir, 'eval', chart_type)
        os.mkdir(to_path_train)
        os.mkdir(to_path_eval)
        count = len(os.listdir(from_path))
        train_count = int(count/(train2eval_ratio+1)*train2eval_ratio)
        eval_count = count - train_count
        rand_train = [i for i in range(train_count)]
        random.shuffle(rand_train)
        rand_eval = [i for i in range(eval_count)]
        random.shuffle(rand_eval)
        for idx, file_name in enumerate(os.listdir(from_path)):
            from_name = os.path.join(from_path, file_name)
            if to_svg:
                if idx < train_count:
                    to_name = os.path.join(to_path_train, ('%03d' % (rand_train[idx] + 1)) + '.svg')
                else:
                    to_name = os.path.join(to_path_eval, ('%03d' % (rand_train[idx-train_count] + 1)) + '.svg')
                cairosvg.svg2svg(url=from_name, write_to=to_name)
            else:
                if idx < train_count:
                    to_name = os.path.join(to_path_train, ('%03d' % (rand_train[idx] + 1)) + '.png')
                else:
                    to_name = os.path.join(to_path_eval, ('%03d' % (rand_train[idx - train_count] + 1)) + '.png')
                cairosvg.svg2png(url=from_name, write_to=to_name, output_width=output_size[0], output_height=output_size[1])
            print(to_name)


def svg2png(in_path, out_path, output_size=(224, 224)):
    if not os.path.isdir(in_path):
        raise NotADirectoryError
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for name in os.listdir(in_path):
        in_name = os.path.join(in_path, name)
        out_name = os.path.join(out_path, name)
        if os.path.isfile(in_name):
            if in_path.find('bar') >= 0:
                y = 0
            elif in_path.find('line') >= 0:
                y = 1
            elif in_path.find('pie') >= 0:
                y = 2
            elif in_path.find('scatter') >= 0:
                y = 3
            else:
                y = -1
            cairosvg.svg2png(url=in_name, write_to=out_name.split(name.split('.')[-1])[0]+'png',
                             output_width=output_size[0], output_height=output_size[1])
        else:
            svg2png(in_name, out_name, output_size)
