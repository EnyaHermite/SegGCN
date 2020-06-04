function iou = getiou(labels,confusion)


for l = 1:numel(labels)
    id = labels(l);
    tp = confusion(id,id); % true positive
    fn = sum(confusion(id,:)) - tp; % false negative
    not_ignored = setdiff(labels,id);
    fp = sum(confusion(not_ignored,id)); % false positive
    iou(l,1) = tp/(tp+fn+fp+eps);    
end