function colors = getLabelColor(pt,label)

colors = pt;
classes = {'ceiling', 'floor', 'wall',  'beam', 'column', 'window',...
           'door','table', 'chair','sofa','bookcase', 'board','clutter'};
NumCls = length(classes);
       
load('labelcolor_labelid');
color_labelId = double(color_labelId(2:41,1:3)); % get labels in the correct range 1~40
map = [28,2,1,3,22,33,8,14,16,24,17,30,6];
color_labelId = color_labelId(map,1:3);% use the colors from scannet to colorize the s3dis predictions
for k = 1:NumCls
    index = (label==(k-1));
    if sum(index)>0
        colors(index,1) = color_labelId(k,1);
        colors(index,2) = color_labelId(k,2);
        colors(index,3) = color_labelId(k,3);
    end
end
