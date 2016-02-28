function normData = normalizedData(Data)
m=size(Data,1); %number of training examples
Xi=size(Data,2); % number of inputs

% Now I will normalize all of the variables.

W=zeros(size(Data));

for i=1:Xi
	ma=max(Data(:,i));
	mi=min(Data(:,i));
	ra=ma-mi;
	for j=1:m
		normData(j,i)= (Data(j,i)-mi)/ra;
	end
end

end
