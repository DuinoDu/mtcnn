clear all;

img = imread('test2.jpg');
imshow(img);
hold on;

load bb1
x1 = total_boxes(:,1);
y1 = total_boxes(:,2);
x2 = total_boxes(:,3);
y2 = total_boxes(:,4);
for i = 1:size(x1,1)
    rectangle('Position',[x1(i) y1(i) (x2(i)-x1(i)) (y2(i)-y1(i))], 'LineWidth', 1, 'EdgeColor', 'g')
end

