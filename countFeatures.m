frameSize = 24;
features = 5;
% All five feature types:
feature = [[2,1]; [1,2]; [3,1]; [1,3]; [2,2]];

count = 0;
% Each feature:
for ii = 1:features
    sizeX = feature(ii,1);
    sizeY = feature(ii,2);
    % Each position:
    for x = 0:frameSize-sizeX
        for y = 0:frameSize-sizeY
            % Each size fitting within the frameSize:
            for width = sizeX:sizeX:frameSize-x
                for height = sizeY:sizeY:frameSize-y
                    count=count+1;
                end
            end
        end
    end
end

display(count)