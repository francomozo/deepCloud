from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


class Colors:
    """ Colors
        Three ways of getting the colors:
            1) Colors.{colorname}
            2) Colors.mixed_palette['{colorname}']
            3) Colors.palette['{subpalette}']['{colorname}']
        Subpalettes:
            'grays', 'oranges', 'blues', 'greens', 'purples'
    """
    
    turquoise = '#1ABC9C'
    emerald = '#2ECC71'
    peterRiver = '#3498DB'
    lightPeterRiver = '#6DB5E4'
    amethyst = '#9B59B6'
    wetAsphalt = '#24495E'
    greenSea = '#16A085'
    nephritis = '#27AE60'
    belizeHole = '#2980B9'
    wisteria = '#8E44AD'
    greenSeaB = '#2C3E50'
    sunFlower = '#F1C40F'
    carrot = '#E67E22'
    alizarin = '#E74C3C'
    clouds = '#ECF0F1'
    concrete = '#95A5A6'
    orange = '#F39C12'
    pumpkin = '#D35400'
    pomegranate = '#C0392B'
    silver = '#BDC3C7'
    asbestos = '#7F8C8D'
    
    mixed_palette = {
        'turquoise' : '#1ABC9C',
        'emerald' : '#2ECC71',
        'peterRiver' : '#3498DB',
        'lightPeterRiver' : '#6db5e4',
        'amethyst' : '#9B59B6',
        'wetAsphalt' : '#24495E',
        'greenSea' : '#16A085',
        'nephritis' : '#27AE60',
        'belizeHole' : '#2980B9',
        'wisteria' : '#8E44AD',
        'greenSeaB' : '#2C3E50',
        'sunFlower' : '#F1C40F',
        'carrot' : '#E67E22',
        'alizarin' : '#E74C3C',
        # 'clouds' : '#ECF0F1', # too light
        'concrete' : '#95A5A6', 
        'orange' : '#F39C12',
        'pumpkin' : '#D35400',
        'pomegranate' : '#C0392B',    
        'silver' : '#BDC3C7',
        'asbestos' : '#7F8C8D'
    }
    
    palette = {
        'grays' : {
            'silver' : '#BDC3C7',
            'asbestos' : '#7F8C8D',
            'concrete' : '#95A5A6'
        },
        # 'whites' : {
        #     'clouds' : '#ECF0F1',
        # },
        'oranges' : {
            'pumpkin' : '#D35400',
            'pomegranate' : '#C0392B',
            'orange' : '#F39C12',
            'sunFlower' : '#F1C40F',
            'carrot' : '#E67E22',
            'alizarin' : '#E74C3C'
        },
        'blues' : {
            'belizeHole' : '#2980B9',
            'peterRiver' : '#3498DB',
            'lightPeterRiver' : '#6db5e4',
        },
        'greens' : {
            'greenSeaB' : '#2C3E50',
            'greenSea' : '#16A085',
            'nephritis' : '#27AE60',
            'wetAsphalt' : '#24495E',
            'turquoise' : '#1ABC9C',
            'emerald' : '#2ECC71'
        },
        'purples' : {
            'wisteria' : '#8E44AD',
            'amethyst' : '#9B59B6'
        }
        
    }
    
    def get_names():
        return ['turquoise', 'emerald', 'peterRiver', 'lightPeterRiver', 'amethyst', 'wetAsphalt', 'greenSea', 
                'nephritis', 'belizeHole', 'wisteria', 'greenSeaB', 'sunFlower', 'carrot',
                'alizarin', 'concrete', 'orange', 'pumpkin', 'pomegranate',
                'silver', 'asbestos']
    
    def get_subpalette_names(subpalette):
        return list(Colors.palette[subpalette].keys())
        
    def print_palette():
        pprint(Colors.palette)
        
    def view_colors():
        x = np.arange(0, 10, .1)
        y = np.ones_like(x)

        plt.figure(figsize=(15,9))
        plt.axis('off')
        plt.title('Color palette', family='serif', fontsize=20)
        for index, (color_name, color) in enumerate(Colors.mixed_palette.items()):
            plt.plot(x, y + index, color=color, linewidth=24)
            s = color_name #+ '=' + "" + color + ""
            plt.text(0, index+.85, s, family='serif', fontsize=15)
            
    def random_color(del_subpalettes=None, use_subpalettes=None):
        """

        Args:
            del_subpalettes ([list], optional): List with strings of the subpalette 
                                            names not to use. Defaults to None.
            use_subpalettes ([list], optional): List with strings of the subpalette 
                                            names to use. Defaults to None.
        """
        
        # case 1: use all colors
        if use_subpalettes is None and del_subpalettes is None:
            color_choice = np.random.choice(Colors.get_names())
            return Colors.mixed_palette[color_choice]
        
        # case 2: use only use_subpaletes
        elif use_subpalettes is not None and del_subpalettes is None:
            
            palettes = [Colors.palette[subpalette].items() for subpalette in use_subpalettes]
            new_palette =  {key: val for palette in palettes for (key, val) in palette}
            color_choice = np.random.choice(list(new_palette.keys()))
            return Colors.mixed_palette[color_choice]
        
        # case 3: dont use del_subpalettes
        elif use_subpalettes is None and del_subpalettes is not None:
            
            new_use_subpalettes = list(Colors.palette.keys())
            new_use_subpalettes = [palette_item for palette_item in new_use_subpalettes if palette_item not in del_subpalettes] 
            palettes = [Colors.palette[subpalette].items() for subpalette in new_use_subpalettes]
            
            new_palette =  {key: val for palette in palettes for (key, val) in palette}
            color_choice = np.random.choice(list(new_palette.keys()))
            return Colors.mixed_palette[color_choice]
        
        # case 4: raise error
        else:
            raise ValueError('use_subpalettes and del_subpalettes cannot be used simultaneously.')
            
    
# =============================================================================

class Linestyles:
    solid   = 'solid'
    doted   = 'dotted'
    dashed  = 'dashed'
    dashdot = 'dashdot'    
    
    loosely_dotted = (0, (1, 10))
    densely_dotted = (0, (1, 1))
    
    loosely_dashed = (0, (5, 10))
    densely_dashed = (0, (5, 1))

    loosely_dashdotted = (0, (3, 10, 1, 10))
    densely_dashdotted = (0, (3, 1, 1, 1))

    loosely_dashdotdotted = (0, (3, 10, 1, 10, 1, 10))
    densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
    
    def view_linestyles():
        linestyle_tuple = [
        ('loosely dotted',        (0, (1, 10))),
        ('dotted',                (0, (1, 1))),
        ('densely dotted',        (0, (1, 1))),

        ('loosely dashed',        (0, (5, 10))),
        ('dashed',                (0, (5, 5))),
        ('densely dashed',        (0, (5, 1))),

        ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),

        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        linestyles = linestyle_tuple[::-1]
        
        X, Y = np.linspace(0, 100, 10), np.zeros(10)
        yticklabels = []

        for i, (name, linestyle) in enumerate(linestyles):
            ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5, color='black')
            yticklabels.append(name)

        ax.set(xticks=[], ylim=(-0.5, len(linestyles)-0.5),
            yticks=np.arange(len(linestyles)), yticklabels=yticklabels)

        # For each line style, add a text annotation with a small offset from
        # the reference point (0 in Axes coords, y tick value in Data coords).
        for i, (name, linestyle) in enumerate(linestyles):
            ax.annotate(repr(linestyle),
                        xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                        xytext=(-6, -12), textcoords='offset points',
                        color="blue", fontsize=8, ha="right", family="monospace")
