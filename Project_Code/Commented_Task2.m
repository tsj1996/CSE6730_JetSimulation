Mesh = readgri('a3_mesh5.gri');
[IE, BE] = EdgeHashMy(Mesh);

Ga = 1.4;
a = 1*pi/180;
M_inf = 2.2;
n_ite = 2000;
CFL = 1;

%initialization
U = zeros(4,Mesh.nElem);
U_inf = zeros(4,Mesh.nElem);
dt_Ai = zeros(1,Mesh.nElem);
R_L1_tot = zeros(1,n_ite);
ATPR = zeros(1,n_ite);


%initial condition
U_free = [1; M_inf*cos(a); M_inf*sin(a);(1/((Ga-1)*Ga))+(M_inf^2)/2];
for i = 1:Mesh.nElem
    U_inf(:,i) = [1; M_inf*cos(a); M_inf*sin(a);(1/((Ga-1)*Ga))+(M_inf^2)/2];
    U(:,i) = [1; M_inf*cos(a); M_inf*sin(a);(1/((Ga-1)*Ga))+(M_inf^2)/2];
end


for i = 1:n_ite
    R = zeros(4,Mesh.nElem); %residual
    s_dl = zeros(1,Mesh.nElem); %????
    R_L1_tot(i) = 0;  %L1 Residual
    for j = 1:length(BE)
        if BE(j,4) == 1      %Engine, wall
            v_plus = [U(2,BE(j,3))/U(1,BE(j,3)); U(3,BE(j,3))/U(1,BE(j,3))];  %2x1 v = [rho*u/rho,rho*v/rho]
            v_b = v_plus-v_plus'*get_N_vec(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:))*get_N_vec(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:));
            v_n = v_plus'*get_N_vec(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:))*get_N_vec(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:));
            p_b = (Ga-1)*(U(4,BE(j,3))-0.5*U(1,BE(j,3))*(norm(v_b)^2));
            p = (Ga-1)*(U(4,BE(j,3))-0.5*U(1,BE(j,3))*norm(v_plus)^2);
            %c = sqrt(Ga*p/U(1,BE(j,3)));
            c = sqrt((Ga-1)*( U(4,BE(j,3))/U(1,BE(j,3))-0.5*norm(v_plus)^2));

            Fhat = [0; p_b.*get_N_vec(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:)); 0];
            smax = norm(v_n)+c;
            dl = get_dl(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:));
            R(:,BE(j,3)) = R(:,BE(j,3))+Fhat.*dl;
            s_dl(:,BE(j,3)) = s_dl(:,BE(j,3))+smax*dl;

        elseif BE(j,4) == 4       %Inflow
            [Fhat,smax] = Roe_Flux(U(:,BE(j,3)),U_inf(:,1),get_N_vec(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:)));
            dl = get_dl(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:));
            R(:,BE(j,3)) = R(:,BE(j,3))+Fhat.*dl;
            s_dl(:,BE(j,3)) = s_dl(:,BE(j,3))+abs(smax)*dl;
        else
            [Fhat,smax] = Roe_Flux(U(:,BE(j,3)),U(:,BE(j,3)),get_N_vec(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:)));
            dl = get_dl(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:));
            R(:,BE(j,3)) = R(:,BE(j,3))+Fhat.*dl;
            s_dl(:,BE(j,3)) = s_dl(:,BE(j,3))+smax*dl;
         end
    end
    
    for j = 1:length(IE)
         N_vec = get_N_vec(Mesh.Node(IE(j,1),:),Mesh.Node(IE(j,2),:));
         [Fhat, smax] = Roe_Flux(U(:,IE(j,3)),U(:,IE(j,4)), N_vec);
         dl = get_dl(Mesh.Node(IE(j,1),:),Mesh.Node(IE(j,2),:));
         R(:,IE(j,3)) = R(:,IE(j,3))+Fhat.*dl;
         s_dl(:,IE(j,3)) = s_dl(:,IE(j,3))+abs(smax)*dl;

         
         [Fhat, smax] = Roe_Flux(U(:,IE(j,4)),U(:,IE(j,3)), -N_vec);
         dl = get_dl(Mesh.Node(IE(j,1),:),Mesh.Node(IE(j,2),:));
         R(:,IE(j,4)) = R(:,IE(j,4))+Fhat.*dl;
         s_dl(:,IE(j,4)) = s_dl(:,IE(j,4))+abs(smax)*dl;
    end


    
    for j = 1:Mesh.nElem
        dt_Ai(j) = 2*CFL/s_dl(j);
        U(:,j) = U(:,j)-dt_Ai(j)*R(:,j);
        Rj = abs(R(1,j))+abs(R(2,j))+abs(R(3,j))+abs(R(4,j));
        %Rj = norm(R(:,j));
        R_L1_tot(i) = R_L1_tot(i) + Rj;
    end
    
    %ATPR
    pt_free = get_pt(U_free);
    d = 1;
    k = 1;
    for j = 1:length(BE)
        if BE(j,4) == 2
            dl = get_dl(Mesh.Node(BE(j,1),:),Mesh.Node(BE(j,2),:));
            pt = get_pt(U(:,BE(j,3)));
            pt_intg(k,1) = (pt/pt_free)*dl;
            k = k+1;
        end
    end
    
    ATPR(i) = (1/d)*sum(pt_intg);
end

figure (1)
hold on
set(gca,'LineWidth',2.5,'FontSize',22);
xlabel('Iterations')
ylabel('Residual L1 Norm')
title('Residual L1 Norm Convergence')
plot(R_L1_tot, 'LineWidth',2.5)

figure (2)
hold on
set(gca,'LineWidth',2.5,'FontSize',22);
xlabel('Iterations')
ylabel('ATPR')
title('ATPR Convergence')
plot(ATPR, 'LineWidth',2.5)


k = 1;
for i = 1:Mesh.nElem
    for j = 1:3
        point(k,1) = Mesh.Node(Mesh.Elem(i,j),1);
        point(k,2) = Mesh.Node(Mesh.Elem(i,j),2);
        face(i,j) = k;
        color1(k,1) = get_Mach(U(:,i));
        color2(k,1) = get_pt(U(:,i));
        k = k+1;
    end
end

figure(3)
patch('Faces',face,'Vertices',point,'FaceVertexCData',color1, 'FaceColor','flat','EdgeColor','none');
hold on 
set(gca,'LineWidth',2,'FontSize',20);
xlabel('X')
ylabel('Y')
title('Mach Number Field Plot')
colorbar

figure(4)
patch('Faces',face,'Vertices',point,'FaceVertexCData',color2, 'FaceColor','flat','EdgeColor','none');
hold on 
set(gca,'LineWidth',2,'FontSize',20);
xlabel('X')
ylabel('Y')
title('Total Pressure Field Plot')
colorbar


            

