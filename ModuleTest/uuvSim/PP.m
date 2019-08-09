
% After running Simulink, copy the following codes to the Command Window

%%%%%%%%%%%%%%%%%%%%% %%  Coordination  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clf;
axis([-100,800,-100,800]);
xlabel('Y');
ylabel('X');
box on;
n=size(tout,1);
for i=1:n
    cla;
    hold on;
  % plot(positiony1(:,2),positionx1(:,2),'b',n1(:,2),m1(:,2),'--r')
    plot(positiony1(:,2),positionx1(:,2),'b',n1(:,2),m1(:,2),'--r',positiony2(:,2),positionx2(:,2),'k',n2(:,2),m2(:,2),'--r',positiony3(:,2),positionx3(:,2),'c',n3(:,2),m3(:,2),'--r')
  % plot(x1(i,2),y1(i,2),'o',x2(i,2),y2(i,2),'d',x3(i,2),y3(i,2),'>','markersize',16,'MarkerFaceColor','b');% dynamic point
    plot(positiony1(i,2),positionx1(i,2),'o',positiony2(i,2),positionx2(i,2),'*',positiony3(i,2),positionx3(i,2),'>');% dynamic point  
  % plot(positiony1(i,2),positionx1(i,2),'o');% dynamic point  
    frame=getframe(gcf)
    imind=frame2im(frame);
    [imind,cm] = rgb2ind(imind,256);
    if i==1
         imwrite(imind,cm,'test.gif','gif', 'Loopcount',inf,'DelayTime',1e-4);
    else
         imwrite(imind,cm,'test.gif','gif','WriteMode','append','DelayTime',1e-4);
    end
end



% axis([-2,100,-4,30,]);
% xlabel('Y');
% ylabel('X');
% box on;
% n=size(tout,1);
% for i=1:n
%     cla;
%     hold on;
%    % plot(x1(:,2),y1(:,2),x2(:,2),y2(:,2),x3(:,2),y3(:,2))% the solid line
%     plot(sqrt((x1(i,2)-x2(i,2))^2),sqrt((x1(i,2)-x2(i,2))^2),'0');
%     %plot(x1(i,2),y1(i,2),'o',x2(i,2),y2(i,2),'*',x3(i,2),y3(i,2),'+');% dynamic point
%     frame=getframe(gcf);
%     imind=frame2im(frame);
%     [imind,cm] = rgb2ind(imind,256);
%     if i==1
%          imwrite(imind,cm,'test.gif','gif', 'Loopcount',inf,'DelayTime',1e-4);
%     else
%          imwrite(imind,cm,'test.gif','gif','WriteMode','append','DelayTime',1e-4);
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Single  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clc;
% clf;
% axis([-2,50,-4,50,]);
% xlabel('Y');
% ylabel('X');
% box on;
% n=size(tout,1);
% for i=1:n
%     cla;
%     hold on;
%     plot(x1(:,2),y1(:,2))% the solid line
%     plot(x1(i,2),y1(i,2),'o');% dynamic point
%     frame=getframe(gcf);
%     imind=frame2im(frame);
%     [imind,cm] = rgb2ind(imind,256);
%     if i==1
%          imwrite(imind,cm,'test.gif','gif', 'Loopcount',inf,'DelayTime',1e-4);
%     else
%          imwrite(imind,cm,'test.gif','gif','WriteMode','append','DelayTime',1e-4);
%     end
% end