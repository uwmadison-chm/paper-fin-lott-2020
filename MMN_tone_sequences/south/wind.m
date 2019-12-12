% from Skuhbinder Kumar

function x=wind(srate,wdms,x);
% srate in Hz, gate duration in ms, vector.
npts=length(x);


 wds=round( 2*wdms/1000 * srate);
 if mod(wds,2)~=0;
     wds=wds+1;
 end

w=linspace(-1*(pi/2),1.5*pi,wds);
w=(sin(w)+1)/2;
x(1:round(wds/2))=x(1:round(wds/2)).*w(1:round(wds/2))';
if(srate==48828)
   x(npts-round(wds/2)+1:npts)=x(npts-round(wds/2)+1:npts).*w(round(wds/2):wds);
else
   x(npts-round(wds/2)+1:npts)=x(npts-round(wds/2)+1:npts).*w(round(wds/2)+1:wds)';
end


