module configuration
!----------------------------------------------!
! Most important parameters and data structures
!----------------------------------------------!
 save

 integer :: lx       
 integer :: ly       
 integer :: nn    ! number of sites   
 integer :: nb    ! number of bonds           
 integer :: mm    ! total string length (mm=ns*ms)
 integer :: nq
 integer :: nq_all
 integer :: ns    ! number of substrings (time slices)
 !integer :: ms    ! maximum sub-string length
 integer :: ntau
 integer :: tstp  
 integer :: bincounter
 integer :: bnd

 integer :: dxl
 integer :: mxl

 integer :: nms1=0
 integer :: nms2=0 
 integer :: nms3=0 
 integer :: nms4=0

 real(8) :: beta  ! inverse temperature
 real(8) :: g  	  ! g=J_{perp}/J
 real(8) :: habs
 real(8) :: qq3
 real(8) :: g2
 real(8) :: jq1
 real(8) :: nu
 real(8) :: prob ! part of the acceptance probability for adding (removing) operator   
 real(8) :: dtau
 real(8) :: sun

 integer :: loopnumt
 integer :: loopnum0
 integer :: loopnum1
 integer :: loopnumper
 integer :: loopnumber
 integer :: loopnummax
 integer :: looporder
 integer :: dloop

 integer :: nh
 integer :: opstchange
 integer, allocatable :: state(:)   
 integer, allocatable :: linktable(:,:)        
 integer, allocatable :: bsites(:,:)  
 integer, allocatable :: opstring(:)  
 integer, allocatable :: tauscale(:)
 integer, allocatable :: oporder(:)  
 integer, allocatable :: opstfp(:,:)  
 integer, allocatable :: rebootloop(:)
 integer :: rebootnum

 integer, allocatable :: frststateop(:) 
 integer, allocatable :: custstateop(:) 
 integer, allocatable :: laststateop(:) 
 integer, allocatable :: vertexlist(:,:) 
 integer, allocatable :: vertexlist_map(:) 
 real(8), allocatable :: rantau(:)  
 real(8), allocatable :: spin(:,:)
 real(8), allocatable :: vec(:,:) 
 integer, allocatable :: phase(:) 
 integer, allocatable :: recorder(:,:)
 integer, allocatable :: recorderhelper(:)

 real(8), parameter :: pi=3.14159265358979323d0

end module configuration

module measurementdata
 !--------------------------------------------!
 !Data we measured
 !--------------------------------------------!
 save

 real(8) :: enrg1=0.d0
 real(8) :: enrg2=0.d0
 real(8) :: amag1=0.d0
 real(8) :: amag2=0.d0
 real(8) :: amag3=0.d0
 real(8) :: amag4=0.d0
 real(8) :: amag5=0.d0
 real(8) :: amag6=0.d0
 real(8) :: amag7=0.d0
 real(8) :: dimr1=0.d0
 real(8) :: dimr2=0.d0
 real(8) :: dimr3=0.d0
 real(8) :: dimerbg=0d0
 real(8) :: stiff=0d0


 integer :: signal=0d0

 real(8), allocatable :: crr(:,:)     	!correlation function in real space

 integer, allocatable :: qpts(:) 	!list of q-points
 integer, allocatable :: tgrd(:)   	!grid of time points
 
 real(8), allocatable :: tcor(:,:)    !correlation function in momentum space
 real(8), allocatable :: tcor_real(:,:)    !correlation function in momentum space
 real(8), allocatable :: tcorpm(:,:)    !correlation function in momentum space
 real(8), allocatable :: tcordm(:,:,:)    !correlation function in momentum space
 real(8), allocatable :: tcordz(:,:,:)    !correlation function in momentum space
 real(8), allocatable :: ref(:,:)     ! real part of fourier transform of states
 real(8), allocatable :: imf(:,:)	  	! imaginary part of fourier transform of states
 real(8), allocatable :: phi(:,:,:)   ! phase factors for fourier transforms
 real(8), allocatable :: tc(:) 
 
 real(8), allocatable :: PMRS_temp(:,:)
 real(8), allocatable :: PMinRealSpace(:,:)
 real(8), allocatable :: demo_or(:,:)
 real(8), allocatable :: demo_pm(:,:)
 real(8), allocatable :: DZinRealSpace(:,:,:)
 real(8), allocatable :: DCinRealSpace(:,:,:)
 integer :: nc

 character(10) :: resname
 character(10) :: vecname
 character(10) :: odpname
 character(10) :: crrname
 character(11) :: tcorname
 character(11) :: tcpmname
 character(11) :: td11name
 character(11) :: tz11name
 character(11) :: td12name
 character(11) :: tz12name
 character(11) :: td21name
 character(11) :: tz21name
 character(11) :: td22name
 character(11) :: tz22name
 character(11) :: deorname
 character(11) :: depmname
 character(11) :: dmbgname

end module measurementdata

!==========================!
!main part of the programme!
!==========================!
! nn     =  number of states
! beta   =  inverse temperature
! dtau   =  time-slize widt
! nbin   =  number of bins
! mstps  =  steps per bin for measurements
! istps  =  initial (equlibration) steps
! gtype  =  time-grid type (1,2,3)
! tmax   =  maximum delta-time for correlation functions
! nqx(y) =  number of q-points for time correlations
! tstp   =  time-step when computing time averages of correlations  
! tmsr   =  measure time correlations after every tmsr MC sweep
! qpts   =  q-values to do (among values 0,....,nn/2, actual q is this times 2*pi/nn)

module vertexupdate
	save
	integer :: v0
	integer :: OpTy0
	integer :: site0
	integer :: site2
 	integer :: ndotz
 	integer, allocatable :: ndotc(:)

	integer :: v1
	integer :: v2
	integer :: b
	integer :: op
	integer :: update_counter
	integer :: loop_counter
	integer :: counter
	integer, allocatable :: dnmsr_counter(:)

	integer :: Spm_measure_signal
	integer :: OpTy1
	integer :: site1
	integer :: vo
	integer :: mm0
	integer :: taumark_o

	integer :: cc
	integer :: cc0
	integer :: cc1

	real(8) :: tau0
	real(8) :: delta_tau
 	real(8) :: dmweight

end module vertexupdate

program Jperp_J_heisenberg_sse

	use configuration
	use measurementdata
	implicit none
	include 'mpif.h'
	integer :: ierr,nprocs
	integer :: rank,sz
	integer :: i,j,nbins,msteps,isteps,tmsr,gtype
	real(8) :: tmax,qq

	open(unit=10, file='read.in', status="old")
	read(10,*)lx,ly,beta,qq3,g2
	read(10,*)sun,bnd
	read(10,*)nbins,msteps,isteps
	read(10,*)dtau
	read(10,*)gtype,tmax              
 	read(10,*)tstp,tmsr 
	close(unit=10)
	jq1=1-qq3
	!jq1=1
	!qq3=qq/dble(1d0-qq)


	!jq1=jq1/sun
	!qq3=qq3/sun
	!2=g2/sun

	!rank=0

	nq=lx/2
	open(unit=10, file='q_resolution.in', status="replace")
	write(10,*)nq
	close(unit=10)
	if (ly>1) then 
		nq_all=6*nq+1
	else
		nq_all=2*nq+1
	endif

	!habs=0d0
	allocate(qpts(nq_all))

 	call MPI_init(ierr)
	call MPI_Comm_size(MPI_COMM_WORLD,nprocs,ierr)
	call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

	call initran(1,rank)
	call makelattice()
	call initconfig(rank)
	call taugrid(gtype,tmax)
	call qarrays()

	print*,"step 1",rank

	prob=beta*nb

!==================================================================!
	!Do isteps equilibration sweeps, find the necessary sweep
	do i=1,isteps
		call Bilibiliupdate(rank,1,10)
		!print*,"a1"
		call adjustcutoff(i)		
		!print*,"b1"
		if (mod(i,5000)==1d0) print*,rank,"warm",i,"/",isteps
		!print*,"c"
	enddo
!==================================================================!

	print*,"step 2",rank


	call writeconfig(rank)
	bincounter=0

	print*,"step 3",rank

	! Do nbins bins with msteps MC sweeps in each, measure after each
	do j=1,nbins
		bincounter=bincounter+1
		do i=1,msteps
			call Bilibiliupdate(rank,1,10)
			!print*,"a2"
			call measure()
			!print*,"b2"
			if (mod(i,tmsr)==0) then
			endif
			if (mod(i,2000)==1d0) print*,rank,"states",i,"/",msteps
			!print*,"e",i
		enddo
		!write bin averages to 'res.dat'!
		print*,bincounter
		call writeresult(rank)
		call writeconfig(rank)
	enddo
	print*,"step 4",rank

	call MPI_Barrier(MPI_COMM_WORLD,ierr)
	call MPI_Finalize(ierr) 

	bincounter=0
	call deallocateall()

end program Jperp_J_heisenberg_sse

!==================================================!

!==================================================!

subroutine initconfig(rank)
	use configuration
	use measurementdata
	implicit none

	integer :: i
	integer :: rank
	real(8), external :: ran

	allocate(state(nn))
 	do i=1,nn
    	state(i)=int(sun*ran())+1
 	enddo

	ns=int(beta/dtau+0.1d0)
	!mm=max(4*ns,nn/4)
	mm=20

!=============================================================================!
	allocate(opstring(0:mm-1))	           !according to whether it is a sub-programme
	opstring(:)=0                         !according to whether it is a sub-programme
	allocate(tauscale(0:mm-1))	           !according to whether it is a sub-programme
	tauscale(:)=0                         !according to whether it is a sub-programme
	allocate(opstfp(0:mm-1,2*bnd))	           !according to whether it is a sub-programme
	opstfp(:,:)=0                         !according to whether it is a sub-programme
	allocate(vertexlist(0:dxl*mm-1,3))        !according to whether it is a sub-programme
	allocate(vertexlist_map(0:dxl*mm-1))    !according to whether it is a sub-programme
	allocate(rantau(nn))				   !according to whether it is a sub-programme
    allocate(oporder(nn))
	nh=0								   !according to whether it is a sub-programme
!=============================================================================!
!=============================================================================!
	!call readconfig(rank)          !according to whether it is a sub-programme
!=============================================================================!	
	allocate(frststateop(nn))
	allocate(custstateop(nn))
	allocate(laststateop(nn))
	allocate(rebootloop(2))
	oporder(:)=-1
	frststateop(:)=-1
	custstateop(:)=-1
	laststateop(:)=-1
	rebootloop(:)=0d0
	rebootnum=0d0
	allocate(crr(int(lx/2)+1,int(ly/2)+1))
	crr(:,:)=0


	allocate(spin(int(sun),2))
	allocate(vec(int(sun),2))
	do i=1,int(sun)
		spin(i,1)=(dble(i-1d0)-dble(sun-1d0)*0.5d0)
		spin(i,2)=-(dble(i-1d0)-dble(sun-1d0)*0.5d0)
		print*,spin(i,:)
		vec(i,1)=cos(2d0*pi/sun*(i-1d0))
		vec(i,2)=sin(2d0*pi/sun*(i-1d0))
		!print*,2d0*pi/sun*i,vec(i,:)
	enddo
	!pause

    vertexlist(:,:)=0
    vertexlist_map(:)=-1
    loopnum0=nn
    loopnum1=0d0
    loopnummax=0d0
    loopnumber=loopnum0+loopnum1

end subroutine initconfig
!==================================================!

!==================================================!

subroutine makelattice()
	use configuration
	implicit none
	integer :: s,x1,x2,y1,y2,s1,s2,s3,s4,s5,s6,t

 	nn=lx*ly

 	if (ly==1) then
    	nb=lx
    	allocate(bsites(2,nb))
    	do x1=0,lx-1
       		s=x1+1
       		x2=mod(x1+1,lx)
       		bsites(1,s)=s
       		bsites(2,s)=x2+1
    	enddo
 		mxl=2
 		dxl=2*mxl

 	elseif (ly==2) then
    	allocate(bsites(2,nb))
    	do y1=0,ly-1
    		do x1=0,lx-1
       			s=1+x1+y1*lx
       			x2=mod(x1+1,lx)
       			y2=y1
       			bsites(1,s)=s
       			bsites(2,s)=1+x2+y2*lx
       			if (y1==0) then
          			x2=x1
          			y2=mod(y1+1,ly)
          			bsites(1,s+nn)=s
          			bsites(2,s+nn)=1+x2+y2*lx       
       			endif
    		enddo
    	enddo
 		mxl=2
 		dxl=2*mxl

 	else
    	nb=(2+2+2)*nn
    	allocate(bsites(2*bnd,nb))
    	do y1=0,ly-1
    		do x1=0,lx-1
       			s=1+x1+y1*lx
       			x2=mod(x1+1,lx)
       			y2=y1
       			bsites(1,s)=s
       			bsites(2,s)=1+x2+y2*lx
       			bsites(3:2*bnd,s)=-1

       			x2=x1
       			y2=mod(y1+1,ly)
       			bsites(1,s+nn)=s
       			bsites(2,s+nn)=1+x2+y2*lx 
       			bsites(3:2*bnd,s+nn)=-1  

       			x2=mod(x1+1,lx)
       			y2=mod(y1+1,ly)
       			bsites(1,s+2*nn)=s
       			bsites(2,s+2*nn)=1+x2+y2*lx
       			bsites(3:2*bnd,s+2*nn)=-1

       			x2=mod(x1+1,lx)
       			y2=mod(ly+y1-1,ly)
       			bsites(1,s+3*nn)=s
       			bsites(2,s+3*nn)=1+x2+y2*lx 
       			bsites(3:2*bnd,s+3*nn)=-1 

       			do t=0,bnd-1
       				s1=1+mod(x1+0,lx)+mod(y1+t,ly)*lx
       				bsites(2*t+1,s+4*nn)=s1
       				s1=1+mod(x1+1,lx)+mod(y1+t,ly)*lx
       				bsites(2*t+2,s+4*nn)=s1

       				s1=1+mod(x1+t,lx)+mod(y1+0,ly)*lx
       				bsites(2*t+1,s+5*nn)=s1
       				s1=1+mod(x1+t,lx)+mod(y1+1,ly)*lx
       				bsites(2*t+2,s+5*nn)=s1
       			enddo
    		enddo
    	enddo
 		mxl=2*bnd
 		dxl=2*mxl
 		nb=2*nn
 	endif

	allocate(phase(nn))
	do s=1,nn
		phase(s)=int(0.5d0*(1+(-1)**(1+mod(s-1,lx)+(s-1)/lx))+1)
		!print*,phase(s)
	enddo
	!pause
	call makelinktype()

end subroutine makelattice
!==================================================!
!==================================================!

subroutine makelinktype()
	use configuration
	implicit none
	integer :: s,x1,x2,y1,y2,s1,s2,s3,s4,s5,s6,t,i

	allocate(linktable(0:dxl-1,0:2))

	do i=0,dxl-1
		linktable(i,0)=mod(i+mxl,dxl)
		linktable(i,1)=ieor(i,1)
		linktable(i,2)=ieor(mod(i+mxl,dxl),1)
	enddo
	!do i=0,dxl-1
	!	print*,"0",i,linktable(i,0)
	!enddo
	!do i=0,dxl-1
	!	print*,"1",i,linktable(i,1)
	!enddo
	!do i=0,dxl-1
	!	print*,"2",i,linktable(i,2)
	!enddo

end subroutine makelinktype
!==================================================!


subroutine Bilibiliupdate(rank,update_type,lim_counter)
	use configuration
	use measurementdata
	implicit none

	integer :: i,j,s,b,op,rank,cs,s1,k,bs,sig,crsign,update_type,lim_counter,update_counter
	integer :: mm_sug,mm_tmp,opod_sug,opod_pre,opod_aft
	real(8) :: wght1,wght2,wght3,wghtq3
	integer, allocatable :: tmststateop(:)
	real(8), external :: ran
	real(8), external :: hf
	allocate(tmststateop(nn))
	tmststateop(:)=0d0
 
 	!if ( rank==0 ) then 
		!do i=1,nb
			!print*,int(hf(i)),i,mod(i-1,lx),int((i-1)/lx),rank
		!enddo
	!endif


	wght1=g2*0.5d0
	wght2=jq1*0.5d0
	wght3=0d0
	wghtq3=qq3*((0.5d0)**3d0)

	update_counter=0d0
	i=mm-1
	!do i=0,mm-1
	!mm_sug=int(mm*ran())

	!print*,"======================================"
	!tmststateop(:)=custstateop(:)
	!do j=1,nh
	!	mm_tmp=oporder(j)
	!	print*,j,mm_tmp,nh
	!	if (opstring(mm_tmp)==0) pause
	!	b=opstring(mm_tmp)/4
	!	call gencurstateop(mm_tmp,b,0)
	!enddo
	!print*,nh+1,oporder(nh+1),nh

	!do j=1,nn
	!	if (custstateop(j)/=tmststateop(j)) then
	!		print*,"miss"
	!	else
	!	!print*,"check"
	!	endif
	!enddo
	!print*,"0000000000000000000000000000000000"

	do
		if (update_type==0) then
			i=mod(i+1,mm)

			if (update_counter>mm) then
				exit
			endif
		elseif (update_type==1) then
			mm_sug=int(mm*ran())
			opod_sug=tauscale(mm_sug)

			if (update_counter>lim_counter) then
				exit
			endif
			!print*,mm_sug,opod_sug,opstring(mm_sug),nh

			if (nh==0) then
			else
				if (opstring(mm_sug)==0) then
					opod_pre=mod(opod_sug+nh-1,nh)+1
				else
					opod_pre=mod(opod_sug+nh-2,nh)+1
				endif
				!print*,"pre",mm_sug,opod_sug,opod_pre,opstring(mm_sug)
				do j=1,opod_pre
					mm_tmp=oporder(j)
					!print*,j,mm_tmp,nh
					if (opstring(mm_tmp)==0) pause
					b=opstring(mm_tmp)/4
					call gencurstateop(mm_tmp,b,0)
				enddo
			endif

			i=mm_sug

		endif
		!print*,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",opstring(mm_sug),nh
		update_counter=update_counter+1
		!print*,i,update_type,update_counter,lim_counter,mm
!=============================================================================================================================!
		op=opstring(i)

		if ( op==0 ) then
			b=int(ran()*nb)+1
			call caldloop(i,b,1)
			if (b<=2*nn) then 
				if ( ran()*(mm-nh)<=(wght2)*prob*(sun**dloop) ) then
					!print*,"add1",sun,dloop,wght2,(wght2)*(sun**dloop),
			 		opstring(i)=4*b
			 		nh=nh+1
			 		call addoperator(i,b,rank)
			 	endif
			 elseif (b>2*nn .and. b<=4*nn) then
				if ( ran()*(mm-nh)<=(wght1)*prob*(sun**dloop) ) then
					!print*,"add1",sun,dloop,(wght1)*prob*(sun**dloop)
			 		opstring(i)=4*b
			 		nh=nh+1
			 		call addoperator(i,b,rank)
			 	endif
			 elseif (b>4*nn) then
			 	if ( ran()*(mm-nh)<=(wghtq3)*prob*(sun**dloop) ) then
					!print*,"add1",sun,dloop,(wghtq3)*prob*(sun**dloop)
			 		opstring(i)=4*b+2
		 			nh=nh+1
			 		call addoperator(i,b,rank)
		 		endif
		 	endif
		elseif ( op/=0 ) then
		 	b=op/4
			call caldloop(i,b,-1)
		 	if (b<=2*nn) then
				if ( ran()*prob*(wght2)<=(mm-nh+1)*(sun**dloop) ) then
			 		opstring(i)=0
			 		opstfp(i,:)=0
		 			nh=nh-1
			 		call remoperator(i,b,rank)
					!call gencurstateop(i,b,0)
			 		!print*,"checkpoint1"
			 	else
					call gencurstateop(i,b,0)
			 		!print*,"checkpoint2"
		 		endif
			elseif (b>2*nn .and. b<=4*nn) then
				if ( ran()*prob*(wght1)<=(mm-nh+1)*(sun**dloop) ) then
			 		opstring(i)=0
			 		opstfp(i,:)=0
		 			nh=nh-1
			 		call remoperator(i,b,rank)
					!call gencurstateop(i,b,0)
			 		!print*,"checkpoint3"
			 	else
					call gencurstateop(i,b,0)
			 		!print*,"checkpoint4"
			 	endif
		 	elseif (b>4*nn) then
				if ( ran()*(wghtq3)*prob<=(mm-nh+1)*(sun**dloop) ) then
			 		opstring(i)=0
			 		opstfp(i,:)=0
		 			nh=nh-1
			 		call remoperator(i,b,rank)
					!call gencurstateop(i,b,0)
			 		!print*,"checkpoint5"
			 	else
					call gencurstateop(i,b,0)
			 		!print*,"checkpoint6"
		 		endif
		 	endif
		 endif
		!print*,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",opstring(mm_sug),nh
!=============================================================================================================================!

		if (update_type==0) then
		elseif (update_type==1) then
			if (nh==0) then
			else
				opod_sug=tauscale(mm_sug)
				opod_aft=mod(opod_sug,nh)+1
				!print*,"aft",mm_sug,opod_sug,opod_aft,opstring(mm_sug)
				if (opod_sug==opod_aft) then
				else
					!print*,opod_sug,mm_sug,nh,"sug"
					do j=opod_aft,nh
						mm_tmp=oporder(j)
						!print*,j,mm_tmp,nh
						if (opstring(mm_tmp)==0) pause
						b=opstring(mm_tmp)/4
						call gencurstateop(mm_tmp,b,0)
					enddo
				endif
			endif
		endif
	enddo

	if (loopnumber/=loopnum0+loopnum1) then
		print*,loopnumber,loopnum0,loopnum1,"neq"
		pause
	endif

	if (update_type==0) then
		do j=1,nh
			mm_tmp=oporder(j)
			b=opstring(mm_tmp)/4
			call gencurstateop(mm_tmp,b,0)
		enddo
	else
		do j=1,nh
			mm_tmp=oporder(j)
			b=opstring(mm_tmp)/4
			call gencurstateop(mm_tmp,b,0)
		enddo
	endif

	do i=1,nn
		if (frststateop(i)/=-1) then
			if (frststateop(i)/=vertexlist_map(custstateop(i))) then
				print*,"boundary lost",frststateop(i),custstateop(i),vertexlist_map(custstateop(i)),&
				&vertexlist_map(frststateop(i)),i
				print*,"boundary lost",int(frststateop(i)/dxl),int(custstateop(i)/dxl),&
				&mod(opstring(int(frststateop(i)/dxl)),2),mod(opstring(int(custstateop(i)/dxl)),2)
				pause
			endif
		else
			if (frststateop(i)/=custstateop(i)) then
				print*,"boundary lost",frststateop(i),custstateop(i),i
				pause
			endif
		endif
	enddo

end subroutine Bilibiliupdate
!==================================================!
!==================================================!
subroutine caldloop(mm_pos,b_pos,crsign)
	use configuration
	use measurementdata
	implicit none

	integer :: i,k,b_pos,s,mm_pos,vt0,vt1,vt2,vt3,crsign,up_pos,leg_pos,j,si,sortmp,l,sortmp2
	integer :: mmt2,mmt3,tt,lp0,mrk1,loopaft,loopbef,loopmid,mrk2,mrk3,mrk4,vtt,l0,l1
	integer, allocatable :: legtag(:,:)
	integer, allocatable :: md(:,:)
	integer, allocatable :: headsort(:,:)
	integer, allocatable :: revtab(:)

	allocate(legtag(0:dxl-1,0:2))
	legtag(:,0)=-1
	legtag(:,1)=0
	legtag(:,2)=-1
	allocate(headsort(0:dxl-1,0:4))
	headsort(:,0)=-1
	headsort(:,1)=0
	headsort(:,2)=-1
	headsort(:,3)=-1
	headsort(:,4)=-1
	allocate(revtab(0:dxl-1))
	revtab(:)=-1
	allocate(md(0:3,2))
	md(:,:)=0d0

	if (crsign==1) then
		if (b_pos<=2*nn ) then
			tt=1
		elseif (b_pos>4*nn .and. b_pos<=6*nn) then
			tt=1
		elseif (b_pos>2*nn .and. b_pos<=4*nn) then
			tt=2
		endif
	elseif (crsign==-1) then
		tt=0
	endif
	loopbef=0d0
	loopaft=0d0
	loopmid=0d0
	si=0d0

	lp0=0d0
	do k=1,mxl
		s=bsites(k,b_pos)
		if (s==-1) cycle
		if (crsign==1) then
			if (custstateop(s)/=-1) then
				vt0=custstateop(s)
				legtag(k-1,1:2)=vertexlist(vt0,1:2)
				vt1=vertexlist_map(vt0)
				!print*,"------add------",int(vt0/dxl),k-1,vertexlist(vt0,1:2),int(vt1/dxl),k-1+mxl,vertexlist(vt1,1:2)
				legtag(k-1+mxl,1:2)=vertexlist(vt1,1:2)
			else
				lp0=lp0+1
				legtag(k-1,1)=-lp0
				legtag(k-1+mxl,1)=-lp0
				legtag(k-1,2)=0
				legtag(k-1+mxl,2)=1
			endif
		elseif (crsign==-1) then
			vt0=dxl*mm_pos+k-1
			legtag(k-1,1:2)=vertexlist(vt0,1:2)
			vt1=dxl*mm_pos+k-1+mxl
			!print*,"------rem------",int(vt0/dxl),k-1,vertexlist(vt0,1:2),int(vt1/dxl),k-1+mxl,vertexlist(vt1,1:2)
			legtag(k-1+mxl,1:2)=vertexlist(vt1,1:2)
		endif
	enddo

	sortmp=0d0
	sortmp2=0d0
	do i=0,dxl-1
		si=bsites(mod(i,mxl)+1,b_pos)
		if (si==-1) cycle
		if (legtag(i,0)==-1) then
			mrk1=legtag(i,1)
			headsort(sortmp,0)=i
			headsort(sortmp,1)=legtag(i,1)
			headsort(sortmp,2)=legtag(i,2)
			headsort(sortmp,3)=sortmp
			revtab(i)=sortmp
			legtag(i,0)=legtag(i,0)+1
			loopbef=loopbef+1

			do j=0,dxl-1
				si=bsites(mod(j,mxl)+1,b_pos)
				if (si==-1) cycle
				mrk2=legtag(j,1)
				if (legtag(j,0)==-1 .and. mrk1==mrk2) then
					l=j
					md(0,1)=l
					md(1,1)=mrk2
					md(2,1)=legtag(l,2)
					md(3,1)=sortmp


					do k=sortmp,dxl-1
						if (headsort(k,1)/=mrk2 .and. headsort(k,1)/=0) then
							print*,"sort error",headsort(k,1),mrk2 
							pause
						endif
						if (headsort(k,2)==-1) then
							headsort(k,:)=md(:,1)
							revtab(l)=k
							sortmp2=k
							exit
						elseif (headsort(k,2)<legtag(j,2)) then
						elseif (headsort(k,2)>legtag(j,2)) then
							md(:,2)=headsort(k,:)
							headsort(k,:)=md(:,1)
							md(:,1)=md(:,2)
							revtab(l)=k
							l=md(0,2)
						endif
					enddo
					legtag(j,0)=legtag(j,0)+1
				endif
			enddo
			sortmp=sortmp2+1
			!print*,sortmp,sortmp2
		endif
	enddo

	do i=0,dxl-1
		if (headsort(i,2)==-1) exit
		l0=headsort(i,0)
		l1=headsort(mod(i+1,dxl),0)
		!print*,"i",i
		if (i==headsort(i,3) .and. &
			&(headsort(headsort(i,3),2)/=1 .or. crsign==-1)) then
			loopmid=loopmid+1
			headsort(i,4)=loopmid
			legtag(l0,0)=legtag(l0,0)+1
		elseif ((headsort(i,3)/=headsort(mod(i+1,dxl),3) .or. mod(i+1,dxl)==headsort(i,3)) .and. &
			&(headsort(headsort(i,3),2)/=1 .or. crsign==-1)) then
			headsort(i,4)=headsort(headsort(i,3),4)
			legtag(l0,0)=legtag(l0,0)+1
		elseif (headsort(i,3)==headsort(mod(i+1,dxl),3)) then
			!print*,"l0,l1",l0,l1,loopmid
			!print*,"con",legtag(:,0)
			!print*,headsort(:,0)
			!print*,"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
			if (legtag(l0,0)==0) then
				vt0=i-headsort(i,3)
				vt1=i+1-headsort(mod(i+1,dxl),3)
				if (vt1==vt0+1 .and. headsort(i,3)==headsort(mod(i+1,dxl),3)) then
					loopmid=loopmid+1
					headsort(i,4)=loopmid
					headsort(mod(i+1,dxl),4)=loopmid
					legtag(l1,0)=legtag(l1,0)+1
					!print*,"add l1",l1,legtag(l1,0)
				else

				endif
				legtag(l0,0)=legtag(l0,0)+1
				!print*,"add l0",l0,legtag(l0,0)
			elseif (legtag(l0,0)==1) then
			endif
		endif
	enddo

	!print*,"00000000000000000000000000000000000000000000000000000"
	!print*,"l0",legtag(:,0)
	!print*,"l1",legtag(:,1)
	!print*,"l2",legtag(:,2)
	!print*,"-----------------------------------------------------"
	!print*,"0",headsort(:,0)
	!print*,"1",headsort(:,1)
	!print*,"2",headsort(:,2)
	!print*,"3",headsort(:,3)
	!print*,"4",headsort(:,4)
	!print*,"rev",revtab(:)
	!print*,"loopbef",loopbef
	
	do i=0,dxl-1
		si=bsites(mod(i,mxl)+1,b_pos)
		if (si==-1) cycle
		!print*,i,legtag(revtab(i),0),headsort(revtab(i),4)
		if (legtag(i,0)==1 .and. (headsort(revtab(i),4)/=0 .and. headsort(revtab(i),4)/=-1)) then
			l0=i
			l1=linktable(l0,tt)

			vt0=headsort(revtab(l0),4)
			vt1=headsort(revtab(l1),4)
			!print*,vt0,vt1
			if (vt0/=vt1) then
				do j=0,dxl-1
					if (headsort(j,4)==vt1) then
						if (j==revtab(l1)) then
							headsort(j,4)=0d0
						else
							headsort(j,4)=vt0
						endif
					endif
					if (headsort(j,4)==vt0) then
						if (j==revtab(l0)) then
							headsort(j,4)=0d0
						else
							headsort(j,4)=vt0
						endif
					endif
				enddo
			elseif (vt0==vt1) then
				loopaft=loopaft+1
				do j=0,dxl-1
					if (headsort(j,4)==vt0) then
						headsort(j,4)=0
					endif
				enddo				
			endif
			!print*,"l0",legtag(:,0)
			!print*,"4",headsort(:,4)
			!print*,"4444444444444444444444444444444",loopaft
		endif
	enddo

	dloop=loopaft-loopbef
	!print*,"loopaft,loopbef,loopmid,dloop,crsign,tt"
	!print*,loopaft,loopbef,loopmid,dloop,crsign,tt
	!print*,"1",headsort(:,1)
	!print*,"2",headsort(:,2)
	!print*,"3",headsort(:,3)
	!print*,"4",headsort(:,4)
	if ((dloop/=1 .and. dloop/=0 .and. dloop/=-1) .or. loopaft==0) then
		!print*,"limit",dloop
	endif

end subroutine caldloop

!==================================================!

!==================================================!
subroutine addoperator(mm_pos,b_pos,rank)
	use configuration
	use vertexupdate
	implicit none
	integer :: i,j,k,mm_pos,b_pos,s0,s1,leg_pos,up_pos,rank
	integer :: vt

	loopnumper=loopnumber
	!print*,"loop",loopnumber,loopnumper
	!print*,"start add"
	call gencurstateop(mm_pos,b_pos,1)
	!print*,"gennet"

	do k=1,mxl
		s0=bsites(k,b_pos)
		if (s0==-1) cycle
		do i=0,1
			leg_pos=k-1+i*mxl
			vt=mm_pos*dxl+leg_pos
			!print*,vt,vertexlist(vt,2),custstateop(s0),frststateop(s0)
			if (vertexlist(vt,3)==2) then 
				!print*,vertexlist(vt,2)
				!print*,"--------------------------------"
				call markloop(vt,0,rank)
			endif
		enddo
	enddo

	if (dloop/=loopnumber-loopnumper) then
		print*,"loopnumber_error_add",loopnumber-loopnumper,dloop,loopnumber,loopnumper,nn
		print*,"loopnumber_error_add",loopnumber-loopnumper,dloop,loopnumt,loopnum1,loopnum0
		pause
	endif
	!print*,"markloop"

	call updateoporder(mm_pos,1)

end subroutine addoperator
!==================================================!

!==================================================!
subroutine remoperator(mm_pos,b_pos,rank)
	use configuration
	use vertexupdate
	implicit none
	integer :: i,j,k,mm_pos,b_pos,s0,s1,leg_pos,up_pos,rank
	integer :: vt

	loopnumper=loopnumber
	!print*,"loop",loopnumber,loopnumper,loopnum1,loopnum0
	!print*,"start rem"
	call gencurstateop(mm_pos,b_pos,-1)
	!print*,"gennet"


	do k=1,mxl
		s0=bsites(k,b_pos)
		if (s0==-1) cycle
		do i=0,1
			leg_pos=k-1+i*mxl
			vt=mm_pos*dxl+leg_pos
			!print*,vertexlist_map(vt),vertexlist(vertexlist_map(vt),2),custstateop(s0),frststateop(s0)
			if (vertexlist_map(vt)/=-1 .and. vertexlist(vertexlist_map(vt),3)==2) then 
				!print*,vertexlist(vt,2)
				!print*,"--------------------------------"
				call markloop(vertexlist_map(vt),0,rank)
			endif
			vertexlist_map(vt)=-1
		enddo
	enddo

	if (dloop/=loopnumber-loopnumper) then
		print*,"loopnumber_error_rem",loopnumber-loopnumper,dloop,loopnumber,loopnumper
		print*,"loopnumber_error_rem",loopnumber-loopnumper,dloop,loopnumt,loopnum1,loopnum0
		pause
	endif
	!print*,"markloop"

	call updateoporder(mm_pos,-1)

end subroutine remoperator
!==================================================!

subroutine markloop(vt,signal,rank)
	use configuration
	use vertexupdate
	implicit none

	integer :: i,j,k,signal,rank,flipsign,mmo,leg0,statechange,tmp,vt
	real(8), external :: ran
	real(8), external :: hf

	if (rebootnum/=0)  then
		loopnumt=rebootloop(1)
		if (rebootnum/=1) then
			rebootloop(1:rebootnum-1)=rebootloop(2:rebootnum)
			rebootloop(rebootnum)=0d0
			rebootnum=rebootnum-1
		else
			rebootloop(1)=0
			rebootnum=rebootnum-1
		endif
	else
		loopnumt=loopnum1+1
	endif

	loopnummax=max(loopnummax,loopnumt)

	Spm_measure_signal=signal
	loop_counter=0d0
	looporder=1

	!do i=1,dxl*mm-1
	!	if (vertexlist(i,1)==loopnumt) then
	!		print*,"mark_error",loopnumt,loopnum1,vertexlist(i,1),rebootnum
	!		print*,"mark_error",vertexlist(i,:)
	!		print*,rebootloop(:)
	!		pause
	!	endif
	!enddo

	v0=vt
	!print*,v0
	!statechange=int(sun*ran())
	statechange=1d0
	cc=statechange

	v1=v0
	counter=1
	if (mod(v0,dxl)>=mxl) then
		cc0=mod(int(sun)-cc,int(sun))				
	else
		cc0=mod(int(sun)+cc,int(sun))			
	endif

	do
		if ( v1==v0 .and. counter/=1 ) then
			!if (Spm_measure_signal==1) 
			!print*,"==============================loop1================================"
			if (statechange/=0d0) loop_counter=loop_counter+1
			loopnumber=loopnumber+1
			loopnum1=loopnum1+1
			!print*,loop_counter,rank
			exit
		endif

		i=v1/dxl
		op=opstring(i)
		b=op/4

		if (v1<0) then
			print*,"v1 error",v1
			pause
		endif
		!if (Spm_measure_signal==1) print*,b,2*nn,v1,v0,rank
		if (b<=2*nn .or. (b>4*nn .and. b<=6*nn)) then
			call turn(i,statechange)
			!print*,"turn"
		else
			call jump(i,statechange)
			!print*,"jump"
		endif
		counter=counter+1

		if ( v2==v0 ) then
			!if (Spm_measure_signal==1) 
			!print*,"==============================loop2================================"
			if (statechange/=0d0) loop_counter=loop_counter+1
			loopnumber=loopnumber+1
			loopnum1=loopnum1+1
			!vertexlist(v2,0)=ieor(vertexlist(v2,0),1)
 			!vertexlist(vertexlist_map(v2),0)=ieor(vertexlist(vertexlist_map(v2),0),1)
			!print*,vertexlist(v2,0)
			!print*,vertexlist(vertexlist_map(v2),0)
			!print*,loop_counter,rank
			exit
		endif
	end do

end subroutine markloop

!==================================================!
subroutine gencurstateop(i,b_pos,crsign)
	use configuration
	use measurementdata
	implicit none

	integer :: i,k,b_pos,s,mm_pos,vt0,vt1,vt2,vt3,crsign,up_pos,leg_pos
	integer :: mmt2,mmt3,rmcounter,stcounter

	mm_pos=i
	if (crsign==0) then
		!print*,"---",crsign,b_pos,opstring(mm_pos)/4
		do k=1, mxl
			s=bsites(k,b_pos)
			if (s==-1) cycle
			vt0=dxl*mm_pos+k-1
			vt1=dxl*mm_pos+k-1+mxl
			if (vertexlist_map(custstateop(s))/=vt0) then
				print*,vertexlist_map(custstateop(s)),vt0,"map_error"
				pause
			endif
			vertexlist_map(custstateop(s))=vt0
			vertexlist_map(vt0)=custstateop(s)
			custstateop(s)=vt1
		enddo
	elseif (crsign==1) then
		!print*,crsign
		do k=1,mxl
			s=bsites(k,b_pos)
			if (s==-1) cycle
			up_pos=0
			leg_pos=k-1+mxl*up_pos
			vt0=mm_pos*dxl+leg_pos
			vt1=mm_pos*dxl+k-1+mxl*(ieor(up_pos,1))
			if (frststateop(s)==-1) then
				frststateop(s)=vt0
				vertexlist_map(vt0)=vt1
				vertexlist_map(vt1)=vt0
				custstateop(s)=vt1
				vertexlist(vt0,3)=2
				vertexlist(vt1,3)=2
				loopnum0=loopnum0-1d0
				loopnumber=loopnumber-1d0
			else
				vt3=custstateop(s)
				vt2=vertexlist_map(custstateop(s))
				!print*,"gen+=========================",loopnumber,loopnumper
				call loadlooptable(vertexlist(vt2,1))
				!print*,"gen00000000000000000000000000",loopnumber,loopnumper
				call loadlooptable(vertexlist(vt3,1))
				!print*,"gen--------------------------",loopnumber,loopnumper

				vertexlist_map(vt3)=vt0
				vertexlist_map(vt0)=vt3

				vertexlist_map(vt2)=vt1
				vertexlist_map(vt1)=vt2

				if (frststateop(s)==vt2) then
					mmt2=int(vt2/dxl)
					mmt3=int(vt3/dxl)
					if (mmt2>mm_pos) then
						frststateop(s)=vt0
					endif
				endif
				custstateop(s)=vt1
				vertexlist(vt0,3)=2
				vertexlist(vt1,3)=2
				vertexlist(vt2,3)=2
				vertexlist(vt3,3)=2
			endif
			!print*,state(s)
		enddo
	elseif (crsign==-1) then
		!print*,crsign
		rmcounter=0d0
		stcounter=0d0
		do k=1,mxl
			s=bsites(k,b_pos)
			if (s==-1) cycle
			stcounter=stcounter+1
			up_pos=0
			leg_pos=k-1+mxl*up_pos
			vt0=mm_pos*dxl+leg_pos
			vt1=mm_pos*dxl+k-1+mxl*(ieor(up_pos,1))
			if (frststateop(s)==vt0 .and. vertexlist_map(vt1)==vt0) then
				frststateop(s)=-1
				vertexlist_map(vt0)=-1
				vertexlist_map(vt1)=-1
				vertexlist(vt0,3)=0
				vertexlist(vt1,3)=0
				custstateop(s)=-1
				!print*,"gen+=========================",loopnumber,loopnumper
				call loadlooptable(vertexlist(vt0,1))
				!print*,"gen00000000000000000000000000",loopnumber,loopnumper
				call loadlooptable(vertexlist(vt1,1))
				!print*,"gen--------------------------",loopnumber,loopnumper
				rmcounter=rmcounter+1
				loopnum0=loopnum0+1d0
				loopnumber=loopnumber+1d0
			else
				vt2=vertexlist_map(vt1)
				vt3=vertexlist_map(vt0)
				!vertexlist_map(vt0)=-1
				!vertexlist_map(vt1)=-1
				custstateop(s)=vt3
				if (frststateop(s)==vt0) then
					frststateop(s)=vt2
				endif
				!print*,"gen+=========================",loopnumber,loopnumper
				call loadlooptable(vertexlist(vt2,1))
				!print*,"gen00000000000000000000000000",loopnumber,loopnumper
				call loadlooptable(vertexlist(vt3,1))
				!print*,"gen--------------------------",loopnumber,loopnumper
				vertexlist_map(vt2)=vt3
				vertexlist_map(vt3)=vt2
				vertexlist(vt0,3)=0
				vertexlist(vt1,3)=0
				vertexlist(vt2,3)=2
				vertexlist(vt3,3)=2
			endif
		enddo
	endif

end subroutine gencurstateop
!==================================================!


!==================================================!
subroutine updateoporder(mm_pos,crsign)
	use configuration
	use measurementdata
	implicit none

	integer :: mm_pos,opod1,opod0,mm_0,mm_1,mm_tmp,opod_tmp1,opod_tmp2
	integer :: i,k,crsign,opod_pos,sz,tmp,ersign
	integer, allocatable :: opordercopy(:)

	i=size(oporder,dim=1)
	if (i<2*nh) then
		allocate(opordercopy(i))
		opordercopy(:)=oporder(:)
		deallocate(oporder)
		allocate(oporder(2*nh))
		oporder(:)=-1
		oporder(1:i)=opordercopy(1:i)
	endif

	if (crsign==1) then
		if (nh==1) then
			tauscale(:)=1
			oporder(1)=mm_pos
		else
			opod0=tauscale(mm_pos)
			opod1=mod(opod0,nh-1)+1
			mm_0=oporder(opod0)
			mm_1=oporder(opod1)

			if (mm_1==mm_0) then
				if (mm_pos<mm_0) then
					oporder(opod0+1:nh)=oporder(opod0:nh-1)
					opod_pos=opod0
					opod0=mod(opod0,nh)+1
				else
					opod_pos=nh
				endif
			elseif (mm_1<mm_0) then
				if (mm_pos>mm_0) then
					opod_pos=nh
				elseif (mm_pos<mm_1) then
					oporder(opod1+1:nh)=oporder(opod1:nh-1)
					opod_pos=opod1
					opod1=mod(opod1,nh)+1
				endif
			else
				oporder(opod1+1:nh)=oporder(opod1:nh-1)
				opod_pos=opod1
				opod1=mod(opod1,nh)+1
			endif
			oporder(opod_pos)=mm_pos


			!print*,"0",opod0,mm_0,mm_pos,nh
			!print*,"1",opod1,mm_1,mm_pos,nh
			!print*,"sta",opod_pos,mm_pos
			!print*,"==============================="
			!pause

			!print*,"++++++++++++++++++++++++++++++++++++++++++"
			!sz=size(oporder,dim=1)
			!print*,"0",opod0,mm_0,mm_pos,nh
			!print*,"1",opod1,mm_1,mm_pos,nh
			!print*,"sta",opod_pos,mm_pos
			!print*,"==============================="
			!tmp=-1
			!ersign=0
			!do i=1,sz
			!	print*,"oporder",i,oporder(i),nh,size(oporder,dim=1)
			!	if (tmp>=oporder(i) .and. i<=nh) then
			!		ersign=1
			!		!pause
			!	endif
			!	tmp=oporder(i)
			!enddo

			if (opod_pos/=nh) then
				do i=opod_pos,nh-1
					opod_tmp1=i
					opod_tmp2=mod(opod_tmp1,nh)+1
					tauscale(oporder(opod_tmp1):oporder(opod_tmp2)-1)=opod_tmp1
				enddo
			endif
			tauscale(oporder(nh):mm-1)=nh
			tauscale(0:oporder(1)-1)=nh
		endif
	elseif (crsign==-1) then
		if (nh==0) then
			tauscale(:)=-1
			oporder(:)=-1
		else
			opod_pos=tauscale(mm_pos)
			opod0=mod(opod_pos-2,nh+1)+1
			opod1=mod(opod_pos,nh+1)+1
			mm_0=oporder(opod0)
			mm_1=oporder(opod1)

			!print*,"]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
			!sz=size(oporder,dim=1)
			!print*,"0",opod0,mm_0,mm_pos,nh
			!print*,"1",opod1,mm_1,mm_pos,nh
			!print*,"sta",opod_pos,mm_pos
			!print*,"==============================="
			!tmp=-1
			!ersign=0
			!do i=1,sz
			!	print*,"oporder",i,oporder(i),nh,size(oporder,dim=1)
			!	if (tmp>=oporder(i) .and. i<=nh) then
			!		ersign=1
			!		!pause
			!	endif
			!	tmp=oporder(i)
			!enddo

			if (opod_pos==nh+1) then
				oporder(opod_pos)=-1
			else
				oporder(opod_pos:nh)=oporder(opod1:nh+1)
				oporder(nh+1)=-1
				!print*,"opod_pos:nh",opod_pos,nh
				!print*,"opod1:nh+1",opod1,nh+1
			endif

			!sz=size(oporder,dim=1)
			!print*,"0",opod0,mm_0,mm_pos,nh
			!print*,"1",opod1,mm_1,mm_pos,nh
			!print*,"sta",opod_pos,mm_pos
			!print*,"==============================="
			!tmp=-1
			!ersign=0
			!do i=1,sz
			!	print*,"oporder",i,oporder(i),nh,size(oporder,dim=1)
			!	if (tmp>=oporder(i) .and. i<=nh) then
			!		ersign=1
			!		!pause
			!	endif
			!	tmp=oporder(i)
			!enddo
			!if (ersign==1) pause

			if (opod0/=nh) then
				do i=opod0,nh-1
					opod_tmp1=i
					opod_tmp2=mod(opod_tmp1,nh)+1
					tauscale(oporder(opod_tmp1):oporder(opod_tmp2)-1)=opod_tmp1
				enddo
			endif
			tauscale(oporder(nh):mm-1)=nh
			tauscale(0:oporder(1)-1)=nh
		endif
	endif


end subroutine updateoporder
!==================================================!


!==================================================!
subroutine loadlooptable(i) 
	use configuration
	use measurementdata
	implicit none

	integer :: i,l,j,k,tmp,tmp2
	integer, allocatable :: rebootloopcopy(:)
	
	!print*,"idx",i
	!print*,"reboot",rebootnum,rebootloop(:)
	!print*,"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
	if (i/=0) then
		rebootnum=rebootnum+1
		l=size(rebootloop,dim=1)
		!print*,"load",size(rebootloop,dim=1),rebootnum,i
		if (l>=rebootnum) then
			tmp=i
			do j=1,rebootnum
				!print*,rebootloop(j),tmp,j,rebootnum,loopnumber
				if (rebootloop(j)==0) then
					rebootloop(j)=tmp
					loopnum1=loopnum1-1
					loopnumber=loopnumber-1
					!print*,"loopnumber=loopnumber-1"
				elseif (rebootloop(j)>tmp) then
					tmp2=rebootloop(j)
					rebootloop(j)=tmp
					tmp=tmp2
				elseif (rebootloop(j)<tmp) then
				elseif (rebootloop(j)==tmp) then
					rebootnum=rebootnum-1
					exit
				endif
			enddo
		else
			allocate(rebootloopcopy(l))
			rebootloopcopy(:)=rebootloop(:)
			deallocate(rebootloop)
			allocate(rebootloop((2*l)))
			rebootloop(:)=0d0
			rebootloop(1:l)=rebootloopcopy(1:l)
			tmp=i
			do j=1,rebootnum
				!print*,rebootloop(j),tmp,j,rebootnum,loopnumber
				if (rebootloop(j)==0) then
					rebootloop(j)=tmp
					loopnum1=loopnum1-1
					loopnumber=loopnumber-1
					!print*,"loopnumber=loopnumber-1"
				elseif (rebootloop(j)>tmp) then
					tmp2=rebootloop(j)
					rebootloop(j)=tmp
					tmp=tmp2
				elseif (rebootloop(j)<tmp) then
				elseif (rebootloop(j)==tmp) then
					rebootnum=rebootnum-1
					exit
				endif
			enddo
		endif

		!print*,"loopnumber",loopnumber
		!print*,"reboot",rebootnum,rebootloop(:)
		!print*,i
		!if (l>nn) pause
	endif

end subroutine loadlooptable
!==================================================!
!==================================================!

subroutine turn(i,statechange)
!-----------------------------------------------------------!
!-----------------------------------------------------------!
 use configuration
 use vertexupdate
 implicit none

 integer :: i,legold,legnew,upold,upnew,opsmold,opsmnew,bstmp
 integer :: temp,statechange,stateupdate
 integer :: s1,s2
 temp=0d0

 opsmold=mod(opstring(i),2)
 legold=mod(v1,dxl)
 upold=int(legold/mxl)
 legold=mod(legold,mxl) 
 bstmp=int(legold/2)

 legnew=ieor(legold,1)
 upnew=upold
 v2=dxl*i+mod(upnew*mxl+legnew,dxl)
 s1=mod(v1,mxl)+1
 s2=mod(v2,mxl)+1
 !print*,s1,s2
 
 vertexlist(v1,1)=loopnumt
 vertexlist(v2,1)=loopnumt 
 vertexlist(v1,2)=looporder
 looporder=looporder+1
 vertexlist(v2,2)=looporder
 looporder=looporder+1
 !print*,"v1",vertexlist(v1,2),"v2",vertexlist(v2,2)
 vertexlist(v1,3)=1
 vertexlist(v2,3)=1
 site2=site1
 site1=-1
 OpTy1=-1
 v1=vertexlist_map(v2)
 if (vertexlist_map(v1)/=v2) then
 	print*,"map error",vertexlist_map(v1),v2,v1
 	pause
 endif
 
update_counter=update_counter+1
 
 !if (Spm_measure_signal==1) print*,"turn",i

 end subroutine turn

!======================================================++++++++++++!

subroutine jump(i,statechange)
!-----------------------------------------------------------!
!-----------------------------------------------------------!
 use configuration
 use vertexupdate
 implicit none

 integer :: i,legold,legnew,upold,upnew,opsmold,opsmnew,bstmp
 integer :: temp,statechange,stateupdate
 integer :: s1,s2
 temp=0d0

 opsmold=mod(opstring(i),2)
 legold=mod(v1,dxl)
 upold=int(legold/mxl)
 legold=mod(legold,mxl) 
 bstmp=int(legold/2)

 legnew=ieor(legold,1)
 upnew=ieor(upold,1)
 v2=dxl*i+mod(upnew*mxl+legnew,dxl)
 s1=mod(v1,mxl)+1
 s2=mod(v2,mxl)+1
 !print*,s1,s2
 
 vertexlist(v1,1)=loopnumt
 vertexlist(v2,1)=loopnumt
 vertexlist(v1,2)=looporder
 looporder=looporder+1
 vertexlist(v2,2)=looporder
 !print*,"v1",vertexlist(v1,2),"v2",vertexlist(v2,2)
 looporder=looporder+1
 vertexlist(v1,3)=1
 vertexlist(v2,3)=1
 site2=site1
 site1=-1
 OpTy1=-1
 v1=vertexlist_map(v2)
update_counter=update_counter+1
 
 !if (Spm_measure_signal==1) print*,"turn",i

 end subroutine jump
!==================================================!

subroutine adjustcutoff(step)
	use configuration
	implicit none

	integer, allocatable :: tauscalecopy(:)
	integer, allocatable :: stringcopy(:)
	integer, allocatable :: opstfpcopy(:,:)
	integer, allocatable :: vertexlistcopy(:,:)
	integer, allocatable :: vertexlist_mapcopy(:)
 	integer :: mmnew,step

 	mmnew=nh+nh/3
 	!print*,"adj"
 	if (mmnew<=mm) return

 	allocate(stringcopy(0:mm-1))
 	stringcopy(:)=opstring(:)
 	deallocate(opstring)
 	allocate(opstring(0:mmnew-1))
 	opstring(:)=0
 	!print*,"check0"

 	allocate(tauscalecopy(0:mm-1))
 	tauscalecopy(:)=tauscale(:)
 	deallocate(tauscale)
 	allocate(tauscale(0:mmnew-1))
 	tauscale(0:mm-1)=tauscalecopy(:)
 	tauscale(mm:mmnew-1)=nh
 	!print*,"check1"

 	allocate(opstfpcopy(0:mm-1,2*bnd))
 	opstfpcopy(:,:)=opstfp(:,:)
 	deallocate(opstfp)
 	allocate(opstfp(0:mmnew-1,2*bnd))
 	opstfp(:,:)=0d0
 	!print*,"check2"

 	opstring(0:mm-1)=stringcopy(:)
 	opstring(mm:mmnew-1)=0
 	deallocate(stringcopy)
 	opstfp(0:mm-1,:)=opstfpcopy(:,:)
 	opstfp(mm:mmnew-1,:)=0
 	deallocate(opstfpcopy)
 	!print*,"check3"

 	allocate(vertexlistcopy(0:dxl*mm-1,3))
 	vertexlistcopy(:,:)=vertexlist(:,:)
 	deallocate(vertexlist)
 	allocate(vertexlist(0:dxl*mmnew-1,3))
 	vertexlist(:,:)=0d0
 	vertexlist(0:dxl*mm-1,:)=vertexlistcopy(0:dxl*mm-1,:)
 	deallocate(vertexlistcopy)
 	!print*,"check4"

 	allocate(vertexlist_mapcopy(0:dxl*mm-1))
 	vertexlist_mapcopy(:)=vertexlist_map(:)
 	deallocate(vertexlist_map)
 	allocate(vertexlist_map(0:dxl*mmnew-1))
 	vertexlist_map(:)=-1
 	vertexlist_map(0:dxl*mm-1)=vertexlist_mapcopy(0:dxl*mm-1)
 	deallocate(vertexlist_mapcopy)
 	!print*,"check5"

 	mm=mmnew

 	open(unit=10,file='info.dat',position='append')
 	write(10,*)' Step: ',step,'  Cut-off L: ',mm,'  Current nh: ',nh
 	close(10)
 	!print*,"check6"

 end subroutine adjustcutoff
!==================================================!
!==================================================!
subroutine measure()
	use configuration
	use measurementdata
	implicit none

	integer :: i,b,op,s0,s1,s2,rx,ry,x,y,ss(0:1),bs,dv,k,sig,j,st
	real(8) :: am1,am2,am3,am4,dm1,dm2,am,ap,dm1t,dm2t
	real(8) :: amt,apt,sp0,sp1
	integer :: st0,st1,mrk0,mrk1,mrk2,looptmp
	real(8) :: sn1,sn2
	real(8), allocatable :: statelist(:)
	real(8), allocatable :: jj(:)
	real(8), allocatable :: ag(:)
 	real(8), external :: ran
	allocate(jj(2))
	allocate(ag(2))

	looptmp=loopnummax
	allocate(statelist(looptmp))
	do i=1, looptmp
		statelist(i)=int(ran()*sun)+1
	enddo

	ag(:)=0d0
	jj(:)=0d0
	ss(:)=0d0

 	 sn2=0d0
 	 do i=1,int(sun)
 	 	sn2=sn2+spin(i,1)*spin(i,1)
 	 	sn1=sn1+spin(i,1)*spin(i,2)
 	 enddo
 	 sn1=sn1/(sun)
 	 sn2=sn2/(sun)

	dm1=0d0
	dm2=0d0
	do i=1,nn		
		s0=i
		if (custstateop(s0)/=-1) then
			mrk0=vertexlist(custstateop(s0),1)
			st0=statelist(mrk0)
		else
			st0=int(ran()*sun)+1
		endif

		ag(1)=ag(1)+vec(st0,1)
		ag(2)=ag(2)+vec(st0,2)

		j=mod(mod(i-1+1,lx),lx)+mod((i-1)/lx,ly)*lx+1
		if (custstateop(j)/=-1) then
			st1=statelist(vertexlist(custstateop(j),1))
			dm1=dm1+spin(st0,phase(s0))*spin(st1,phase(j))
		else
			st1=int(ran()*sun)+1
			dm1=dm1+spin(st0,phase(s0))*spin(st1,phase(j))
		endif
		j=mod(mod(i-1,lx),lx)+mod((i-1)/lx+1,ly)*lx+1
		if (custstateop(j)/=-1) then
			st1=statelist(vertexlist(custstateop(j),1))
			dm2=dm2+spin(st0,phase(s0))*spin(st1,phase(j))
		else
			st1=int(ran()*sun)+1
			dm2=dm2+spin(st0,phase(s0))*spin(st1,phase(j))
		endif
	enddo
	!print*,"am"

	 !am=am/2
	 !ap=ap/2
	 am1=0.d0
	 am2=0.d0
	 am3=0d0
	 am4=0d0

	 do i = 0, mm-1
	 	op=opstring(i)
	 	if (op/=0) then
	 		b=op/4
			call gencurstateop(i,b,0)
			if (b<=2*nn) then
				s0=i*dxl+mxl-1
				mrk0=vertexlist(s0,1)
				st0=statelist(mrk0)
				!print*,mrk0,st0
				s0=bsites(1,b)
    			if (b<=nn) then
    				jj(1)=jj(1)-spin(st0,phase(s0))
    			elseif (b>nn .and. b<=2*nn) then
    				jj(2)=jj(2)-spin(st0,phase(s0))
    			endif
    		endif

	 	endif

	 	am=0d0
	 	ap=0d0
	 	do j=1,nn
	 		if (custstateop(j)/=-1) then
	 			mrk1=vertexlist(custstateop(j),1)
	 			st1=statelist(mrk1)
	 			!print*,st1,mrk1,loopnumber,looptmp,custstateop(j)
	 			!print*,rebootloop(:)
	 			am=am+spin(st1,phase(j))*(-1)**(mod(j-1,lx)+(j-1)/lx)
	 			ap=ap+spin(st1,phase(j))
	 		else
	 			st1=int(ran()*sun)+1
	 			!print*,st1,mrk1,loopnumber,looptmp,custstateop(j)
	 			!print*,rebootloop(:)
	 			am=am+spin(st1,phase(j))*(-1)**(mod(j-1,lx)+(j-1)/lx)
	 			ap=ap+spin(st1,phase(j))
	 		endif
	 	enddo

        am1=am1+abs(dble(am))
    	am2=am2+(dble(am))**2
        am3=am3+dble(ap)
    	am4=am4+(dble(ap))**2

	 end do
	 

	 if (mm/=0) then
    	am1=am1/dble(mm)
    	am2=am2/dble(mm)
    	ag=ag/dble(mm)
    	am3=am3/dble(mm)
    	am4=am4/dble(mm)
 	else
   		am1=dble(abs(am))
    	am2=dble(am)**2
   		am3=dble(ap)
    	am4=dble(ap)**2
 	endif

	 enrg1=enrg1+dble(nh)
 	 enrg2=enrg2+dble(nh)**2
 	 amag1=amag1+am1
 	 amag2=amag2+am2
 	 amag3=amag3+am3
 	 amag4=amag4+am4
 	 amag5=amag5+sqrt(ag(1)**2+ag(2)**2)
 	 amag6=amag6+ag(1)
 	 amag7=amag7+ag(2)
 	 dimr1=dimr1+(dm1)**2
 	 dimr2=dimr2+(dm2)**2
 	 dimr3=dimr3+(dm1)**2+(dm2)**2
 	 nms1=nms1+1
 	 stiff=stiff+0.5d0*(dble(jj(1))**2+dble(jj(2))**2)
 	 !print*,jj(1),jj(2),0.5d0*(dble(jj(1))**2+dble(jj(2))**2)

 	 do s1=1, nn
   		x=mod(s1-1,lx)
   		y=int((s1-1)/lx)
   		mrk1=vertexlist(custstateop(s1),1)
   		do rx=0, lx/2
   			do ry=0, ly/2
     			s2=mod(x+rx,lx)+(mod(y+ry,ly))*lx+1
   				mrk2=vertexlist(custstateop(s2),1)
     			if (mrk1==mrk2) then
     				crr(rx+1,ry+1)=crr(rx+1,ry+1)+sn2*((-1)**phase(s1))*((-1)**phase(s2))
     				if (rx==0 .and. ry==0) then
     					!print*,sn2,phase(s1)*phase(s2),sn2*phase(s1)*phase(s2)
     				endif
     			endif
   			end do
   		end do
 	end do
end subroutine measure
!==================================================!
!==================================================!

subroutine writeresult(rank)
	use configuration
	use measurementdata
	use vertexupdate
	implicit none

	integer :: msteps,rx,ry,q,t,i1,i2,i3,rank,i

	real(8), external :: qx
	real(8), external :: qy

    resname='res000.dat'
    vecname='vec000.dat'
    odpname='odp000.dat'
    crrname='crr000.dat'
    tcorname='tcor000.dat'
    tcpmname='tcpm000.dat'
    td11name='td11000.dat'
    tz11name='tz11000.dat'
    td12name='td12000.dat'
    tz12name='tz12000.dat'
    td21name='td21000.dat'
    tz21name='tz21000.dat'
    td22name='td22000.dat'
    tz22name='tz22000.dat'
    deorname='deor000.dat'
    depmname='depm000.dat'
    dmbgname='dmbg000.dat'

    i3=rank/100
    i2=mod(rank,100)/10
    i1=mod(rank,10)

    resname(6:6)=achar(48+i1)
    resname(5:5)=achar(48+i2)
    resname(4:4)=achar(48+i3)

    vecname(6:6)=achar(48+i1)
    vecname(5:5)=achar(48+i2)
    vecname(4:4)=achar(48+i3)

    odpname(6:6)=achar(48+i1)
    odpname(5:5)=achar(48+i2)
    odpname(4:4)=achar(48+i3)

    crrname(6:6)=achar(48+i1)
    crrname(5:5)=achar(48+i2)
    crrname(4:4)=achar(48+i3)

    tcorname(7:7)=achar(48+i1)
    tcorname(6:6)=achar(48+i2)
    tcorname(5:5)=achar(48+i3)

    tcpmname(7:7)=achar(48+i1)
    tcpmname(6:6)=achar(48+i2)
    tcpmname(5:5)=achar(48+i3)

    td11name(7:7)=achar(48+i1)
    td11name(6:6)=achar(48+i2)
    td11name(5:5)=achar(48+i3)

    tz11name(7:7)=achar(48+i1)
    tz11name(6:6)=achar(48+i2)
    tz11name(5:5)=achar(48+i3)

    td11name(7:7)=achar(48+i1)
    td11name(6:6)=achar(48+i2)
    td11name(5:5)=achar(48+i3)

    tz11name(7:7)=achar(48+i1)
    tz11name(6:6)=achar(48+i2)
    tz11name(5:5)=achar(48+i3)

    td12name(7:7)=achar(48+i1)
    td12name(6:6)=achar(48+i2)
    td12name(5:5)=achar(48+i3)

    tz12name(7:7)=achar(48+i1)
    tz12name(6:6)=achar(48+i2)
    tz12name(5:5)=achar(48+i3)

    td21name(7:7)=achar(48+i1)
    td21name(6:6)=achar(48+i2)
    td21name(5:5)=achar(48+i3)

    tz21name(7:7)=achar(48+i1)
    tz21name(6:6)=achar(48+i2)
    tz21name(5:5)=achar(48+i3)

    td22name(7:7)=achar(48+i1)
    td22name(6:6)=achar(48+i2)
    td22name(5:5)=achar(48+i3)

    tz22name(7:7)=achar(48+i1)
    tz22name(6:6)=achar(48+i2)
    tz22name(5:5)=achar(48+i3)

    deorname(7:7)=achar(48+i1)
    deorname(6:6)=achar(48+i2)
    deorname(5:5)=achar(48+i3)

   	depmname(7:7)=achar(48+i1)
    depmname(6:6)=achar(48+i2)
    depmname(5:5)=achar(48+i3)

    dmbgname(7:7)=achar(48+i1)
    dmbgname(6:6)=achar(48+i2)
    dmbgname(5:5)=achar(48+i3)

 	enrg1=enrg1/dble(nms1)
 	enrg2=enrg2/dble(nms1)
 	amag1=amag1/dble(nms1)
 	amag2=amag2/dble(nms1)
 	amag3=amag3/dble(nms1)
 	amag4=amag4/dble(nms1)
 	amag5=amag5/dble(nms1)
 	amag6=amag6/dble(nms1)
 	amag7=amag7/dble(nms1)
 	stiff=stiff/dble(nms1)
 	dimr1=dimr1/dble(nms1)
 	dimr2=dimr2/dble(nms1)
 	dimr3=dimr3/dble(nms1)

 	enrg2=(enrg2-enrg1*(enrg1+1.d0))/nn
 	enrg1=-(enrg1/(beta*dble(nn)))

 	amag1=amag1/dble(nn)
 	amag2=amag2/dble(nn)**2
 	amag3=amag3/dble(nn)
 	amag4=amag4/dble(nn)**2
 	amag5=amag5/dble(nn)
 	amag6=amag6/dble(nn)
 	amag7=amag7/dble(nn)
 	dimr1=dimr1/dble(nn)**2
 	dimr2=dimr2/dble(nn)**2
 	dimr3=dimr3/dble(nn)**2

 	open(10,file=resname,position='append')
 	write(10,*)enrg1,enrg2,amag1,amag3
 	close(10)

 	open(10,file=odpname,position='append')
 	write(10,*)amag2,amag4,dimr1,dimr2,dimr3
 	close(10)

 	open(10,file=vecname,position='append')
 	write(10,*)amag5,amag6,amag7,stiff
 	close(10)
 
 	enrg1=0.d0
 	enrg2=0.d0
 	amag1=0.d0
 	amag2=0.d0
 	amag3=0.d0
 	amag4=0.d0
 	amag5=0.d0
 	amag6=0.d0
 	amag7=0.d0
 	stiff=0.d0
 	dimr1=0.d0
 	dimr2=0.d0
 	dimr3=0.d0

 	crr=crr/dble(nn)/dble(nms1)
 	open(20,file=crrname,position='append')
 	do rx=0,lx/2
		do ry=0,ly/2
   			write(20,*)rx,ry,crr(rx+1,ry+1)
		enddo
 	end do
 	close(20)
 	crr(:,:)=0.d0

 	nms1=0

 	nms2=0d0

end subroutine writeresult
!==================================================!



!==================================================!

real(8) function ran()
!----------------------------------------------!
! 64-bit congruental generator                 !
! iran64=oran64*2862933555777941757+1013904243 !
!----------------------------------------------!
 implicit none

 real(8) :: dmu64
 integer(8) :: ran64,mul64,add64
 common/bran64/dmu64,ran64,mul64,add64

 ran64=ran64*mul64+add64
 ran=0.5d0+dmu64*dble(ran64)

 end function ran


Subroutine initran(w,rank)
!--------------------------------------------------!
 implicit none
 integer(8) :: irmax
 integer(4) :: w,b
 integer :: rank,system_time
 real(8) :: dmu64
 integer(8) :: ran64,mul64,add64
 common/bran64/dmu64,ran64,mul64,add64
 
irmax=2_8**31
irmax=2*(irmax**2-1)+1
mul64=2862933555777941757_8
add64=1013904243
dmu64=0.5d0/dble(irmax)
call system_clock(system_time)
ran64=abs(system_time-(rank*1993+2018)*20160514)  

end Subroutine initran
!==================================================!

!==================================================!

subroutine deallocateall()
	use configuration
	use measurementdata
	implicit none

 	deallocate(state)
 	!print*,"state"
 	deallocate(bsites)
 	!print*,"bsites"
 	deallocate(opstring)
 	!print*,"opstring"
 	deallocate(opstfp)
 	!print*,"opstfp"
 	deallocate(tauscale)
 	!print*,"tauscale"
 	deallocate(frststateop)
 	!print*,"frststateop"
 	deallocate(laststateop)
 	!print*,"laststateop"
 	deallocate(vertexlist)
 	!print*,"vertexlist"
 	deallocate(vertexlist_map)
 	!print*,"vertexlist_map"
 	deallocate(rantau)
 	!print*,"rantau"
 	deallocate(tc)
 	!print*,"tc"
 	deallocate(phi)
 	!print*,"phi"
 	deallocate(tcor)
 	!print*,"tcor"
 	deallocate(ref)
 	!print*,"ref"
 	deallocate(imf)
 	!print*,"imf"
 	deallocate(crr)
 	!print*,"crr"

end subroutine deallocateall
!==================================================!			

!==================================================!		

subroutine taugrid(gtype,tmax)
!-------------------------------------------------------------------------!
!-constructs array tgrd(0,...,ntau) of time separations (in units of dtau)!
!-gtype = 1 for uniform grid of all separatiins up to tmax
!-        2 for quadratic grid t=dtau*n^2/4 up to beta/2
!-        3 for uniform grid up to tmax followed by quadratic to beta/2
!-actual time points including dtau factor are written to 'tgrid.dat'
!--------------------------------------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: i,t1,t2,gtype,nsm
 real(8) :: tmax

 nsm=int((tmax+1.d-5)/dtau)
 open(10,file='tgrid.dat',status='replace')

 if (gtype==1) then
    ntau=nsm
    write(10,'(i4)')ntau
    allocate(tgrd(0:ntau)) 
    do i=0,ntau
       tgrd(i)=i
       write(10,'(f15.8)')tgrd(i)*dtau
    enddo

 elseif (gtype==2) then    
    ntau=0
    t1=0
    do
       t2=((ntau+1)**2)/4
       if (t2==t1) t2=t1+1
       if (t2.le.nsm) then
          ntau=ntau+1
          t1=t2
       else
          exit
       endif
    enddo
    write(10,'(i4)')ntau
    allocate(tgrd(0:ntau)) 
    tgrd(0)=0
    write(10,'(f15.8)')0.d0
    do i=1,ntau
       tgrd(i)=(i**2)/4
       if (tgrd(i)==tgrd(i-1)) tgrd(i)=tgrd(i-1)+1       
       write(10,'(f15.8)')tgrd(i)*dtau
    enddo

 elseif (gtype==3) then  
    ntau=nsm
    t1=0
    i=0
    do
       t2=((i+1)**2)/4
       if (t2==t1) t2=t1+1
       if (t2.gt.nsm.and.t2.le.ns/2) then
          ntau=ntau+1
          t1=t2
       elseif (t2.gt.ns/2) then
          exit
       endif
       i=i+1
    enddo
    write(10,'(i4)')ntau
    allocate(tgrd(0:ntau)) 
    tgrd(0)=0
    do i=0,nsm
       tgrd(i)=i
       write(10,'(f15.8)')tgrd(i)*dtau
    enddo
    ntau=nsm
    t1=0
    i=0
    do
       t2=((i+1)**2)/4
       if (t2==t1) t2=t1+1
       if (t2.gt.nsm.and.t2.le.ns/2) then
          ntau=ntau+1
          tgrd(ntau)=t2
          write(10,'(f15.8)')tgrd(ntau)*dtau
          t1=t2
       elseif (t2.gt.ns/2) then
          exit
       endif
       i=i+1
    enddo
 endif 
 close(10)

 allocate(tc(0:ntau)) 
 allocate(tcor_real(0:ntau,nn)) 
 allocate(tcor(0:ntau,nq_all)) 
 allocate(tcorpm(0:ntau,nq_all))
 allocate(tcordm(0:ntau,nn,4))
 allocate(tcordz(0:ntau,nn,4))
 allocate(PMinRealSpace(nn,0:ntau))
 allocate(PMRS_temp(nn,0:ntau))
 allocate(DZinRealSpace(nn,0:ntau,4))
 allocate(DCinRealSpace(nn,0:ntau,4))
 allocate(demo_pm(nn,0:ntau))
 allocate(demo_or(nn,0:ntau))
 tcor=0.d0
 tcorpm=0d0
 tcordz=0d0
 nms3=0d0
 PMRS_temp=0d0
 PMinRealSpace=0d0
 demo_pm=0d0
 demo_or=0d0
 nc=0d0
 DZinRealSpace=0d0
 DCinRealSpace=0d0

 end subroutine taugrid
 !==================================================!		



!==================================================!

 subroutine random_tau()
!---------------------------------------------!
!time dependent q-space correlation functions S+S-
!---------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: i,j,k
 real(8), external :: ran

 if ( size(rantau, dim=1)/=nh+1 ) then
 	deallocate(rantau)
 	allocate(rantau(nh))
 endif 

 rantau(nh)=beta
 do i=1,nh
 	rantau(i)=ran()*beta
 enddo

 call quick_sort(rantau,nh,1,nh)
 
 end subroutine random_tau
!==================================================!
!==================================================!

recursive subroutine quick_sort(a,n,s,e)
implicit none
integer :: n      
real(8) :: a(1:n+1) 
integer :: s      
integer :: e      
integer :: l,r    
real(8) :: k      
real(8) :: temp   
l=s
r=e+1
if ( r<=l ) return

k=a(s) 
do while(.true.)

	do while( .true. )
		l=l+1
		if ( (a(l) > k) .or. (l>=e) ) exit
	end do

	do while( .true. )
		r=r-1
		if ( (a(r) < k) .or. (r<=s) ) exit
	end do
	if ( r <= l ) exit

	temp=a(l)
	a(l)=a(r)
	a(r)=temp
end do
 
temp=a(s)
a(s)=a(r)
a(r)=temp

call quick_sort(a,n,s,r-1) 
call quick_sort(a,n,r+1,e)
return
end subroutine quick_sort
!==================================================!

subroutine measure2()
!---------------------------------------------!
! time dependent q-space correlation functions
!---------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: i,j,rx,ry,r,s,q,b,op,s1,s2,t1,t2,dt,dd,s0,taumark,dv,k,sig,bs
 real(8), external :: qx
 real(8), external :: qy
 real(8) :: dimer_background
 integer, allocatable :: dimer_config(:,:)
 real(8), allocatable :: dimer_space(:,:)
 allocate(dimer_config(ns,nn))
 allocate(dimer_space(nn,0:ntau))
 dimer_space=0d0
 dimer_config=0d0
 dimer_background=0d0


 i=0
 s0=1
 dd=0d0
 do i=0,mm-1
    op=opstring(i)
    taumark=tauscale(i)
	do s=s0,ns
    	if ( dble(s)/dble(ns)*beta > rantau(mod(taumark,nh)+1) ) then
    		s0=s
    		exit 
    	endif

    enddo
 enddo

 dimer_background=dimer_background/dble(dd)

 do q=1,nn           ! loop over q points
    dd=0
    tc=0d0
    do t1=1,ns,tstp   ! average time point t1 over time slices
       	do dt=0,ntau
       	enddo
       	dd=dd+1        ! count the number of measurements for t-averages
    enddo
    tcor(:,q)=tcor(:,q)+tc(:)/dble(dd)  
 enddo

 dimerbg=dimerbg+dimer_background

 nms2=nms2+1


end subroutine measure2
 
!==================================================!
	
subroutine qarrays()
!-----------------------------------------------------------!
! phase factors for fourier transforms of state configuration
!-----------------------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: r,q
 real(8), external :: qx
 real(8), external :: qy
 real(8) :: qqx,qqy

 allocate(phi(2,nn,nq_all))
 allocate(ref(ns,nq_all))
 allocate(imf(ns,nq_all))

 if (ly>1) then
 	do q=1, nq_all
    	qqx=2.d0*pi*dble(qx(q-1))/dble(lx)
    	qqy=2.d0*pi*dble(qy(q-1))/dble(ly)
    	do r=1,nn
       		phi(1,r,q)=cos(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn))       
       		phi(2,r,q)=sin(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn)) 
    	enddo
 	enddo
  else
 	do q=1, nq_all
    	qqx=2.d0*pi*dble(q-1)/dble(lx)
    	qqy=2.d0*pi*dble(0)/dble(ly)
    	do r=1,nn
       		phi(1,r,q)=cos(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn))       
       		phi(2,r,q)=sin(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn)) 
    	enddo
 	enddo
  endif


 end subroutine qarrays
!----------------------!

real(8) function qx(q)
!----------------------------------------------!
!-find qx point-!
!----------------------------------------------!
 use configuration
 use measurementdata

 implicit none

 integer :: q

 if ( int(q/nq) == 0 ) then
	qx=mod(q,nq)
 elseif(int(q/nq) == 1) then
	qx=nq 
 elseif(int(q/nq) == 2) then
	qx=nq-mod(q,nq)
 elseif(int(q/nq) == 3) then
	qx=0
 elseif(int(q/nq) == 4) then
	qx=mod(q,nq)
 elseif(int(q/nq) == 5) then
	qx=nq-mod(q,nq)
 elseif(int(q/nq) == 6) then
	qx=0
 endif

 end function qx

 !----------------------------------------------!
 !----------------------------------------------!
 !----------------------------------------------!

 real(8) function qy(q)
!----------------------------------------------!
!-find qy point-!
!----------------------------------------------!
 use configuration
 use measurementdata

 implicit none

 integer :: q

 if (int(q/nq) == 0) then
	qy=mod(q,nq)
 elseif(int(q/nq) == 1) then
	qy=nq-mod(q,nq)
 elseif(int(q/nq) == 2) then
	qy=0
 elseif(int(q/nq) == 3) then
	qy=mod(q,nq)
 elseif(int(q/nq) == 4) then
	qy=nq
 elseif(int(q/nq) == 5) then
	qy=nq-mod(q,nq)
 elseif(int(q/nq) == 6) then
	qy=0
 endif

 end function qy

!----------------------------------------------!
!----------------------------------------------!

subroutine writeconfig(rank)
!--------------------------------------!
 use configuration 
 implicit none
 integer :: i,rank
 character(len=3) :: ranks
 write (ranks,fmt='(i3.3)') rank
 open(10,file='conf'//ranks//'.log',status='replace')
 do i=1,nn
    write(10,'(i2)')state(i)
 enddo
 write(10,*)mm,nh
 do i=0,mm-1
    write(10,*)opstring(i),opstfp(i,:)
 enddo
 do i=0,dxl*mm-1
    write(10,*)vertexlist_map(i),vertexlist(i,1),vertexlist(i,2),vertexlist(i,3)
 enddo
 write(10,*)loopnum0,loopnum1,loopnumber
 close(10)
 end subroutine writeconfig
!------------------------!

 subroutine readconfig(rank)
!---------------------!
 use configuration
 implicit none
 integer :: i,rank,j,op
 character(len=3) :: ranks
 write (ranks,fmt='(i3.3)') rank
 print*, 'conf'//ranks//'.log'
 open(10,file='conf'//ranks//'.log',status='old')
 do i=1,nn
    read(10,*)state(i)
 enddo
 read(10,*)mm,nh
 allocate(opstring(0:mm-1))
 allocate(opstfp(0:mm-1,2*bnd))
 allocate(tauscale(0:mm-1))
 allocate(vertexlist(0:dxl*mm-1,3))
 allocate(vertexlist_map(0:dxl*mm-1))
 allocate(rantau(nh)) 
 allocate(oporder(2*nh))
 do i=0,mm-1
    read(10,*)opstring(i),opstfp(i,:)
 enddo
 tauscale(:)=0
 do i=0,dxl*mm-1
    read(10,*)vertexlist_map(i),vertexlist(i,1),vertexlist(i,2),vertexlist(i,3)
 enddo
 read(10,*)loopnum0,loopnum1,loopnumber
 close(10)

 j=nh
 do i=0,mm-1
   	op=opstring(i)
   	if (op==0) then
   		tauscale(i)=j
   		cycle
   	endif
    j=mod(j,nh)+1
    oporder(j)=i
    tauscale(i)=j
 enddo
 end subroutine readconfig

!----------------------------------------------!
!----------------------!