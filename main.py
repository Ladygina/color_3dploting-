import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from scipy.interpolate import interp1d

def data_random_generate():
    r = np.linspace(0, 1.25, 50)
    p = np.linspace(0, 2 * np.pi, 50)
    r_list =[]

    X = r * np.cos(p)  # preparing arrays
    Yc = []
    Z = np.linspace(0, 10, 20)
    X, Zc = np.meshgrid(X, Z)
    Xc = []

    for i in range(len(Z)):
        if i % 5 != 0:  # random condition for test
            r = np.linspace(i, 1.25 + i, 50)
            p = np.linspace(i, 2 * np.pi + i, 50)
        else:
            r = np.linspace(0, 15, 50)
            p = np.linspace(0, 2 * np.pi, 50)
        r_list.append(r)
        X, Y = r * np.cos(p), r * np.sin(p)
        Xc.append(X)
        Yc.append(Y)

    return X, Y, Xc, Yc, Z, Zc,  r_list

def read_numbers_from_file(filename, quantity):
    with open(filename, 'r') as file:
        # Считываем весь файл и разбиваем строки на числа
        data = file.read()
        numbers = list(map(float, data.split()))

    # Проверяем, будет ли размер numbers меньше quantity
    for i in range(0, len(numbers), quantity):
        yield numbers[i:i + quantity]
    return numbers
# Используем функцию
filename = '1.txt'  # Укажите свой файл
quantity = 50  # Количество чисел, которое хотите считать за раз

layers_DNA =[]
for numbers_chunk in read_numbers_from_file(filename, quantity):
    print(numbers_chunk)
    layers_DNA.append(numbers_chunk)
a=1
def insert(dic, keyvalue, value):
    if keyvalue in dic:
        dic[keyvalue].append(value)
    else:
        dic[keyvalue] = []
        dic[keyvalue].append(value)
def gradient_definition(r_list, z_list):
    max_list = []
    min_list = []

    max_map ={}
    list_color = []

    for list_num in range(len(r_list)):
        max_curr = np.max(r_list[list_num])

        min_curr = np.min(r_list[list_num])

        max_list.append(max_curr)  # список индексов максимальных элементов по слоям
        min_list.append(min_curr)  # список индексов минимальных элементов по слоям

        insert(max_map, list_num, max_curr)   # словарь ключ - индекс слоя, значение - максимальное значение диаграммы направленности

    max_list.sort()   # сортировка максимумов по возрастанию
    max_list = list(set(max_list))   # удаление дубликатов

    colours = cm.rainbow(np.linspace(0, 1, len(max_list)))

    # rgb format
    color_map_max ={}
    for i in range( len(max_list)):
        color_map_max[max_list[i]] = colours[i]     # словарь ключ - максимальное значение диаграммы направленности,
                                                    # значение - цвет
    for i in range(len(z_list)):
        key_color = max_map[i][0]
        list_color.append( color_map_max[key_color])   # список цветов по слоям

    return list_color



def plot_surface(data_load_flag):
    if data_load_flag:
        Z = np.linspace(0, 20, len(layers_DNA))
        p = np.linspace(0, 2 * np.pi, len(layers_DNA[0]))
        Xc =[]
        Yc =[]

        for i in range(len(layers_DNA)):
            X = layers_DNA[i]* np.cos(p)
            Y = layers_DNA[i]* np.sin(p)
            Xc.append(X)
            Yc.append(Y)
        X, Zc = np.meshgrid(X, Z)
        fig = plt.figure(figsize=(7, 4))
        ax_3d = fig.add_subplot(projection='3d')
        custom_color = gradient_definition(layers_DNA, Z)  # RGB
        custom_cmap = matplotlib.colors.ListedColormap(custom_color)  # facecolor
        ax_3d.plot_surface(Xc, Yc, Zc, alpha=0.8, rstride=1, cstride=1, cmap=custom_cmap)
        #ax_3d.plot_wireframe(Xc, Yc, Zc, rstride=1, cstride=1, color='black')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        plt.show()

        return layers_DNA, X, Y, Xc, Yc, Zc, Z, custom_cmap, custom_color
    else:
        print('generating data:')
        X, Y, Xc, Yc, Z, Zc,  r_list = data_random_generate()

        fig = plt.figure(figsize=(7, 4))
        ax_3d = fig.add_subplot(projection='3d')
        custom_color = gradient_definition(r_list,Z)     #RGB
        custom_cmap = matplotlib.colors.ListedColormap(custom_color)   # facecolor
        ax_3d.plot_surface(Xc, Yc, Zc, alpha=0.8, rstride=1, cstride=1, cmap= custom_cmap)
        ax_3d.plot_wireframe(Xc, Yc, Zc, rstride=1, cstride=1,  color = 'black')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        plt.show()

        return r_list, X, Y, Xc, Yc, Zc, Z, custom_cmap, custom_color
def color_approximation( start_color, end_color, n ):
    t = np.linspace(0, 1,n)

    # Вычисляем промежуточные значения цветов
    r = (1 - t) * start_color[0] + t * end_color[0]   # градиент R
    g = (1 - t) * start_color[1] + t * end_color[1]   # градиент G
    b = (1 - t) * start_color[2] + t * end_color[2]   # градиент B

    # Преобразуем значения цветов в диапазон от 0 до 1
    r = r / 255
    g = g / 255
    b = b / 255
    return t, r,g,b

def contour_approximation(r_list, z_list, num_step):
    r_size = len( r_list[0] )
    x_approximate = []
    y_approximate = []
    z_approximate = []
    approx_cmap = []
    for i in range(1, len(z_list)):    # количесвто векторов по слоям
        for j in range(r_size):        # количесвто точек ДНА

            first_point_r = r_list[i-1][j]
            second_point_r = r_list[i][j]

            first_point_z = z_list[i-1]
            second_point_z = z_list[i]


            R = [ first_point_r, second_point_r]
            Z = [first_point_z, second_point_z]

            f = interp1d(R, Z)
            r_new = np.linspace(first_point_r, second_point_r, num_step)
            z_new = f(r_new)


            x_new = np.cos(phi[j])*r_new
            y_new = np.sin(phi[j])*r_new

            x_approximate.append(list(x_new))
            y_approximate.append(list(y_new))
        z_approximate.append(list(z_new))

        color_approx_curr = cm.rainbow(np.linspace(0, 1, len(z_new)))
        approx_cmap.append (color_approx_curr)

    return x_approximate, y_approximate, z_approximate, approx_cmap


def color_list_transform(color_list):
    list_rgb =[]
    for i in range(len(color_list)):
        curr_arr = color_list[i]*255
        curr_arr = curr_arr[:3]
        list_rgb.append(curr_arr)
    return list_rgb


######################-------------------------------------
def main_call(step, phi, data_load_flag):
    r_list,X, Y, Xc, Yc, Zc, z_list, custom_cmap, custom_color = plot_surface(data_load_flag)

    x_approx, y_approx , z_approx, approx_vec = contour_approximation(r_list, z_list,step)   # разбиение координат с шагом step
    x_array = np.array(x_approx)
    y_array = np.array(y_approx)
    z_array = np.array(z_approx)

    x_array = np.reshape(x_array, (step*( len(z_list)-1)*len(X))) if data_load_flag==0 \
                                                                  else np.reshape(x_array, (step*( len(z_list)-1)*len(layers_DNA[0])))
    # изменение размерности для построения поверхности
    y_array = np.reshape(y_array, (step*( len(z_list)-1)*len(X))) if data_load_flag==0 \
                                                                  else np.reshape(y_array, (step*( len(z_list)-1)*len(layers_DNA[0])))
    z_array = np.reshape(z_array, step*len(z_approx))

    ####################################

    list_rgb = color_list_transform(custom_color)

    fig = plt.figure(figsize=(17, 14))
    ax_3d = fig.add_subplot(projection='3d')

    for i in range(len(z_array)):
        if i < (len(z_array)-1):
             t, r_gradient, g_gradient, b_gradient = color_approximation(list_rgb[i//step], list_rgb[i//step+1], step)
             cmap_curr = cm.colors.LinearSegmentedColormap.from_list('custom', list(zip(t, zip(r_gradient, g_gradient, b_gradient))))  # градиент между соседними слоями

             x_curr = x_array[i*len(X): (i+1)*len(X)] if data_load_flag==0  else  x_array[i*len(layers_DNA[0]): (i+1)*len(layers_DNA[0])]
             y_curr = y_array[i*len(X): (i+1)*len(X)] if data_load_flag==0  else  y_array[i*len(layers_DNA[0]): (i+1)*len(layers_DNA[0])]
             ax_3d.scatter(x_curr, y_curr, c = x_curr, zs = z_array[i], s=500, marker='s' , cmap=cmap_curr)   # отрисовка

    ax_3d.plot_wireframe(Xc, Yc, Zc, alpha=0.8, rstride=1, cstride=1, color='black')                          # отрисовка контура
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    plt.show()

phi = np.linspace(0, 2*np.pi, 50)
step = 4
main_call(step, phi, 1)