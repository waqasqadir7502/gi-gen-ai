import "./pagename.css" 
export default function PageName({ name} :{ name: string}){
    return(
        <div className="page-name-bar flex justify-center items-center">     
        <div className="page-name-inner">
        <div className="page-name">
          <h2>{name}</h2>
          <p>
            Home. Pages. <span>{name}</span>
          </p>
          </div>
        </div>
      </div>
    )
}