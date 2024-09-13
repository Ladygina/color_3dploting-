import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

def data_random_generate(N):
    length = 1000
    for i in range(N):
        with open(str(N)+'.txt', 'w') as f:
            max_val = float( np.random.random(1)*1000)
            arr_half1 = np.array( np.linspace(0, max_val , length//2))
            arr_half2 = np.array(np.linspace(max_val, 0, length//2))
            arr_i = np.concatenate((arr_half1, arr_half2))

            f.write(str(arr_i))
    return arr_i

arr = data_random_generate(1)
phi = np.linspace(0, 2 * np.pi, len(arr))


def lat_lot_to_xy(R, lat, lon):          # lat - угол между осью z и Oxy(широта),
    x = R * np.cos(lat) * np.cos(lon)    # lon - угол между проекцией вектора и Oy
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x,y, z



data_random_generate(3)

def read_plot_files(N):
    layout_arrays = []


    fig = plt.figure(layout='constrained')
    ax1 = fig.add_subplot(1, 1, 1, projection='polar', theta_offset=np.pi / 2)
    for i in range(N):
        with open(str(i+1) +'.txt', 'r') as f:
            arr= f.readlines()
        phi = np.linspace(0, 2*np.pi, len(arr))+i
        ax1.plot(phi, arr)
        layout_arrays.append(phi)
        layout_arrays.append(arr)

    plt.show()
    return layout_arrays

layers = read_plot_files(1)

def insert(dic, keyvalue, value):
    if keyvalue in dic:
        dic[keyvalue].append(value)
    else:
        dic[keyvalue] = []
        dic[keyvalue].append(value)
def gradient_definition(r_list, z_list):  # r_list=[[...], [...],...,[...]]
    Max = -10000  # p_list=[[...], [...],...,[...]]
    Min = 10000

    max_list = []
    min_list = []
    max_position = 0
    min_position = 0
    max_map ={}
    list_color = []

    for list_num in range(len(r_list)):
        max_curr = np.max(r_list[list_num])
        max_position = np.argmax(r_list[list_num])

        min_curr = np.min(r_list[list_num])
        min_position = np.argmin(r_list[list_num])

        max_list.append(max_curr)  # список индексов максимальных элементов по слоям
        min_list.append(min_curr)  # список индексов минимальных элементов по слоям

        list_curr=[]
        insert(max_map, list_num, max_curr)

    max_list.sort()   # сортировка максимумов по возрастанию
    max_list = list(set(max_list))   # удаление дубликатов

    colours = cm.rainbow(np.linspace(0, 1, len(max_list)))

    # rgb format
    color_map_max ={}
    for i in range( len(max_list)):
        color_map_max[max_list[i]] = colours[i]


    for i in range(len(z_list)):
        key_color = max_map[i][0]
        list_color.append( color_map_max[key_color])



    return list_color



def plot_surface(data_load_flag):
    if data_load_flag:
        N_layers = len(layers)


        fig = plt.figure(figsize=(7, 4))
        ax_3d = fig.add_subplot(projection='3d')

        for i in range(1,N_layers):
            x_grid = layers[i][0]
            y_grid = layers[i][1]

            z_grid = np.zeros((len(x_grid), len(y_grid)))
            #ax_3d.plot(x_grid, y_grid, x_grid)
            #ax_3d.plot_surface(x_grid, y_grid, z_grid,  rstride=5, cstride=5, cmap='plasma')
       # plt.show()

    else:
        print('generating data:')

        r_list =[]
        z_list =[]
        fig = plt.figure(figsize=(17, 14))
        ax_3d = fig.add_subplot(projection='3d')

        r = np.linspace(0, 1.25 , 50)
        p = np.linspace(0, 2 * np.pi, 50)


        X = r* np.cos(p)                                 # preparing arrays
        Yc = []
        Z = np.linspace(0, 10, 10)
        X,Zc = np.meshgrid(X,Z)
        Xc=[]

        for i in range(len(Z)):
            if i%3!=0:                               # random condition for test
                r = np.linspace(i, 1.25+i, 50)
                p = np.linspace(i, 2 * np.pi+i, 50)
            else:
                r = np.linspace(0, 15, 50)
                p = np.linspace(0, 2 * np.pi, 50)
            r_list.append(r)
            X, Y = r * np.cos(p), r * np.sin(p)

            Xc.append(X)
            Yc.append(Y)

        custom_color = gradient_definition(r_list,Z)     #RGB
        custom_cmap = matplotlib.colors.ListedColormap(custom_color)   # facecolor
        ax_3d.plot_surface(Xc, Yc, Zc, alpha=0.8, rstride=1, cstride=1, cmap= custom_cmap)
        #ax_3d.plot_surface(Xc, Yc, Zc, alpha=0.8, rstride=1, cstride=1)
        ax_3d.plot_wireframe(Xc, Yc, Zc, rstride=1, cstride=1,  color = 'black')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        plt.show()

        return r_list, Xc, Yc, Zc, Z, custom_cmap, custom_color
def color_approximation( start_color, end_color, n ):

    # Задаем начальный и конечный цвет в формате RGB
    # start_color = (255, 0, 0)  # Красный
    # end_color = (0, 0, 255)  # Синий

    # Создаем массив значений от 0 до 1 (для плавного перехода)
    t = np.linspace(0, 1,n)

    # Вычисляем промежуточные значения цветов
    r = (1 - t) * start_color[0] + t * end_color[0]
    g = (1 - t) * start_color[1] + t * end_color[1]
    b = (1 - t) * start_color[2] + t * end_color[2]

    # Преобразуем значения цветов в диапазон от 0 до 1
    r = r / 255
    g = g / 255
    b = b / 255

    # color_gradient =
    # # Создаем фигуру и отображаем градиент
    # fig, ax = plt.subplots(figsize=(10, 2))
    # ax.imshow([np.linspace(0, 1, 100)], aspect='auto',
    #           cmap=plt.cm.colors.LinearSegmentedColormap.from_list('custom', list(zip(t, zip(r, g, b)))))
    # ax.set_axis_off()
    # plt.show()
    return t, r,g,b

def contour_approximation(r_list, phi_list, z_list, color_list, num_step):
    r_size = len( r_list[0] )
    x_approximate = []
    y_approximate = []
    z_approximate = []
    approx_cmap = []
    for i in range(1, len(z_list)):   # количесвто векторов по слоям
        for j in range(r_size):        # количесвто точек ДНА

            first_point_r = r_list[i-1][j]
            second_point_r = r_list[i][j]

            first_point_z = z_list[i-1]
            second_point_z = z_list[i]

            start_color = color_list[i-1]
            end_color = color_list[i]
            t, r, g, b = color_approximation(start_color,end_color, num_step)
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.imshow([np.linspace(0, 1, 100)], aspect='auto',
                       cmap=plt.cm.colors.LinearSegmentedColormap.from_list('custom', list(zip(t, zip(r, g, b)))))
            ax.set_axis_off()
            plt.show()

            R = [ first_point_r, second_point_r]
            Z = [first_point_z, second_point_z]

            f = interp1d(R, Z)
            r_new = np.linspace(first_point_r, second_point_r, num_step)
            z_new = f(r_new)

            x_new = np.cos(phi[j])*np.ones(num_step)*r_new
            y_new = np.sin(phi[j])*np.ones(num_step)*r_new

            x_approximate.append(list(x_new))
            y_approximate.append(list(y_new))
            z_approximate.append(list(z_new))

        color_down = color_list[i-1]
        color_up = color_list[i]

        color_approx_curr = cm.rainbow(np.linspace(0, 1, len(z_new)))
        approx_cmap.append (color_approx_curr)


    return x_approximate, y_approximate, z_approximate, approx_cmap



######################-------------------------------------
r_list, Xc, Yc, Zc, z_list, custom_cmap, custom_color = plot_surface(0)
phi = np.linspace(0, 2*np.pi, 50)
step = 4
x_approx, y_approx , z_approx, approx_vec = contour_approximation(r_list,phi,  z_list,custom_color, step)
x_array = np.array(x_approx)
y_array = np.array(y_approx)
z_array = np.array(z_approx)

x_array = np.reshape(x_array, step*len(x_approx))
y_array = np.reshape(y_array, step*len(x_approx))
z_array = np.reshape(z_array, step*len(x_approx))

x_mesh, z_mesh = np.meshgrid(x_array, z_array)

approx_vec = np.reshape(approx_vec, (len(z_list)-1)*step*step)
approx_cmap = matplotlib.colors.ListedColormap(approx_vec)

fig = plt.figure(figsize=(17, 14))
ax_3d = fig.add_subplot(projection='3d')
#ax_3d.plot_surface(Xc, Yc, Zc, alpha=1, rstride=1, cstride=1, cmap= custom_cmap)
ax_3d.scatter(x_array, y_array, z_array, c=z_array, facecolor=approx_vec)
plt.show()




color_approximation((255, 0,0), (0,0,255), 4)






